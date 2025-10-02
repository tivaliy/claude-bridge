"""FastAPI router for Anthropic Messages API endpoints."""

import logging
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse

from ..core.claude_client import ClaudeClient
from .adapter import AnthropicMessageAdapter
from .schemas import MessagesRequest, MessagesResponse
from .streaming import stream_anthropic_response

logger = logging.getLogger("claude_bridge.router")

router = APIRouter(prefix="/anthropic", tags=["Anthropic API"])


@router.post("/v1/messages", response_model=MessagesResponse)
async def create_message(request: MessagesRequest):
    """
    Create a message using Claude Code CLI.

    Supports both streaming and non-streaming responses.
    Compatible with Anthropic Messages API format.
    Supports images and PDF documents via base64 encoding.
    """
    started = time.perf_counter()
    temp_files: list[Path] = []

    try:
        logger.debug(
            "Incoming /messages request",
            extra={
                "model": request.model,
                "stream": request.stream,
                "messages_count": len(request.messages),
                "max_tokens": request.max_tokens,
            },
        )
        # Convert Anthropic messages to Claude CLI format (with file extraction)
        prompt, system_prompt, temp_files = await AnthropicMessageAdapter.messages_to_prompt(
            request.messages, request.system
        )

        if not prompt:
            raise HTTPException(status_code=400, detail="No user message found in request")

        # Reject tools/function calling (not implemented)
        if request.tools:
            raise HTTPException(
                status_code=400,
                detail="Tools/function calling is not supported. Claude CLI does not support custom tools.",
            )

        # Warn about ignored sampling parameters
        ignored_params = []
        if request.temperature is not None:
            ignored_params.append(f"temperature={request.temperature}")
        if request.top_p is not None:
            ignored_params.append(f"top_p={request.top_p}")
        if request.top_k is not None:
            ignored_params.append(f"top_k={request.top_k}")
        if request.stop_sequences:
            ignored_params.append(f"stop_sequences={request.stop_sequences}")

        if ignored_params:
            logger.warning(
                "Sampling parameters are not supported by Claude CLI and will be ignored",
                extra={"ignored_parameters": ignored_params},
            )

        # Initialize Claude client (CLI wrapper)
        claude_client = ClaudeClient()

        # Execute query via CLI with file references
        # Note: max_tokens is part of Anthropic API but not supported by Claude CLI --print mode
        cli_response = claude_client.query(
            prompt=prompt,
            system_prompt=system_prompt,
            model=request.model,
            stream=request.stream,
            file_paths=temp_files,
        )

        # Handle streaming vs non-streaming
        if request.stream:
            # Return SSE stream (cleanup handled in streaming function)
            logger.debug("Streaming response initiated", extra={"temp_files": len(temp_files)})
            return StreamingResponse(
                stream_anthropic_response(cli_response, request.model, temp_files),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )
        else:
            # Return complete response
            response = await AnthropicMessageAdapter.cli_response_to_anthropic(
                cli_response, request.model
            )
            duration = time.perf_counter() - started
            logger.debug(
                "Completed non-streaming response",
                extra={
                    "duration_sec": round(duration, 4),
                    "model": request.model,
                    "output_tokens": response.usage.output_tokens,
                    "temp_files": len(temp_files),
                },
            )
            return response
    except HTTPException:
        raise
    except Exception as e:
        duration = time.perf_counter() - started
        logger.error(
            "Error handling /messages request",
            exc_info=True,
            extra={
                "model": request.model,
                "stream": request.stream,
                "duration_sec": round(duration, 4),
            },
        )
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)) from e
    finally:
        # Clean up temp files (for non-streaming; streaming handles its own cleanup)
        if not request.stream and temp_files:
            for temp_file in temp_files:
                try:
                    temp_file.unlink(missing_ok=True)
                    logger.debug("Deleted temp file", extra={"file": str(temp_file)})
                except Exception as cleanup_error:
                    logger.warning(
                        "Failed to delete temp file",
                        extra={"file": str(temp_file), "error": str(cleanup_error)},
                    )
