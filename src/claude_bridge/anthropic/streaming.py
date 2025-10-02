"""Streaming support for Anthropic Messages API format."""

import json
import logging
import time
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

logger = logging.getLogger("claude_bridge.streaming")


async def stream_anthropic_response(
    cli_response: AsyncIterator[dict[str, Any]],
    model: str,
    temp_files: list[Path] | None = None,
) -> AsyncIterator[str]:
    """
    Stream Claude CLI response in Anthropic Messages API SSE format.

    CLI streaming JSON format (--output-format stream-json --include-partial-messages):

    With --include-partial-messages (TRUE REAL-TIME STREAMING):
    {
        "type": "stream_event",
        "event": {
            "type": "content_block_delta",
            "delta": {"type": "text_delta", "text": "incremental text"}
        }
    }

    Without flag (LEGACY BUFFERED):
    {
        "type": "assistant",
        "message": {"content": [{"text": "complete text"}]}
    }

    Args:
        cli_response: AsyncIterator from Claude CLI (stream-json format)
        model: Model name from request
        temp_files: List of temporary files to clean up after streaming

    Yields:
        SSE formatted event strings in Anthropic format
    """
    message_id = f"msg_{uuid.uuid4().hex[:24]}"
    content_index = 0
    content_started = False
    output_tokens = 0
    input_tokens = 0
    started = time.perf_counter()

    logger.debug(
        "Begin streaming",
        extra={"model": model, "message_id": message_id, "temp_files": len(temp_files or [])},
    )

    try:
        # Event 1: message_start
        yield "event: message_start\n"
        yield f"data: {json.dumps({
            'type': 'message_start',
            'message': {
                'id': message_id,
                'type': 'message',
                'role': 'assistant',
                'content': [],
                'model': model,
                'stop_reason': None,
                'stop_sequence': None,
                'usage': {'input_tokens': 0, 'output_tokens': 0}
            }
        })}\n\n"

        # Process CLI stream
        async for chunk in cli_response:
            if not isinstance(chunk, dict):
                continue

            chunk_type = chunk.get("type", "")

            # Handle stream_event wrapper (when using --include-partial-messages)
            if chunk_type == "stream_event":
                event = chunk.get("event", {})
                event_type = event.get("type", "")

                # Handle message_start
                if event_type == "message_start":
                    # Already sent in our initial message_start above
                    continue

                # Handle content_block_start
                elif event_type == "content_block_start":
                    if not content_started:
                        content_started = True
                        # Forward the content_block_start event
                        yield "event: content_block_start\n"
                        yield f"data: {json.dumps({
                            'type': 'content_block_start',
                            'index': event.get('index', 0),
                            'content_block': event.get('content_block', {'type': 'text', 'text': ''})
                        })}\n\n"
                    continue

                # Handle content_block_delta - THIS IS THE KEY FOR REAL-TIME STREAMING!
                elif event_type == "content_block_delta":
                    # Send content_block_start before first delta
                    if not content_started:
                        content_started = True
                        yield "event: content_block_start\n"
                        yield f"data: {json.dumps({
                            'type': 'content_block_start',
                            'index': content_index,
                            'content_block': {'type': 'text', 'text': ''}
                        })}\n\n"

                    delta = event.get("delta", {})
                    text = delta.get("text", "")

                    if text:
                        output_tokens += len(text.split())

                        yield "event: content_block_delta\n"
                        yield f"data: {json.dumps({
                            'type': 'content_block_delta',
                            'index': content_index,
                            'delta': {'type': 'text_delta', 'text': text}
                        })}\n\n"

                # Handle content_block_stop, message_delta, etc.
                elif event_type == "content_block_stop":
                    # We'll handle this at the end
                    continue

                elif event_type == "message_delta":
                    usage_data = event.get("usage", {})
                    input_tokens = usage_data.get("input_tokens", input_tokens)
                    output_tokens = usage_data.get("output_tokens", output_tokens)

                continue  # Skip further processing for stream_event types

            # Handle assistant message chunks (legacy format without --include-partial-messages)
            if chunk_type == "assistant":
                message = chunk.get("message", {})
                content = message.get("content", [])

                # First content chunk - send content_block_start
                if not content_started and content:
                    yield "event: content_block_start\n"
                    yield f"data: {json.dumps({
                        'type': 'content_block_start',
                        'index': content_index,
                        'content_block': {'type': 'text', 'text': ''}
                    })}\n\n"
                    content_started = True

                # Send content_block_delta for each text block
                for block in content:
                    if block.get("type") == "text":
                        text = block.get("text", "")
                        if text:
                            # Estimate tokens (rough approximation)
                            output_tokens += len(text.split())

                            yield "event: content_block_delta\n"
                            yield f"data: {json.dumps({
                                'type': 'content_block_delta',
                                'index': content_index,
                                'delta': {'type': 'text_delta', 'text': text}
                            })}\n\n"

                # Extract usage from message
                if "usage" in message:
                    usage = message["usage"]
                    input_tokens = usage.get("input_tokens", input_tokens)
                    output_tokens = usage.get("output_tokens", output_tokens)

            # Handle final result (has final usage info)
            elif chunk_type == "result":
                if "usage" in chunk:
                    usage = chunk["usage"]
                    input_tokens = usage.get("input_tokens", input_tokens)
                    output_tokens = usage.get("output_tokens", output_tokens)

        # Event N-2: content_block_stop
        if content_started:
            yield "event: content_block_stop\n"
            yield f"data: {json.dumps({
                'type': 'content_block_stop',
                'index': content_index
            })}\n\n"

        # Event N-1: message_delta
        yield "event: message_delta\n"
        yield f"data: {json.dumps({
            'type': 'message_delta',
            'delta': {'stop_reason': 'end_turn', 'stop_sequence': None},
            'usage': {'output_tokens': output_tokens}
        })}\n\n"

        # Event N: message_stop
        yield "event: message_stop\n"
        yield f"data: {json.dumps({'type': 'message_stop'})}\n\n"

        duration = time.perf_counter() - started
        logger.debug(
            "Streaming complete",
            extra={
                "model": model,
                "message_id": message_id,
                "duration_sec": round(duration, 4),
                "output_tokens": output_tokens,
                "input_tokens": input_tokens,
            },
        )

    finally:
        # Clean up temp files after streaming completes
        if temp_files:
            for temp_file in temp_files:
                try:
                    temp_file.unlink(missing_ok=True)
                    logger.debug("Deleted temp file", extra={"file": str(temp_file)})
                except Exception as cleanup_error:
                    logger.warning(
                        "Failed to delete temp file",
                        extra={"file": str(temp_file), "error": str(cleanup_error)},
                    )
