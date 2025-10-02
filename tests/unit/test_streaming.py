"""Unit tests for streaming SSE conversion."""

import json

import pytest

from claude_bridge.anthropic.streaming import stream_anthropic_response


class TestStreamAnthropicResponse:
    async def test_stream_event_format_basic(self, make_async_iter):
        cli_chunks = [
            {
                "type": "stream_event",
                "event": {
                    "type": "content_block_delta",
                    "delta": {"type": "text_delta", "text": "Hello"},
                },
            }
        ]

        events = []
        async for event in stream_anthropic_response(
            make_async_iter(cli_chunks), "claude-sonnet-4"
        ):
            events.append(event)

        # Should produce: message_start, content_block_start, delta, content_block_stop,
        # message_delta, message_stop
        event_types = [self._extract_event_type(e) for e in events]
        assert "message_start" in event_types
        assert "content_block_start" in event_types
        assert "content_block_delta" in event_types
        assert "content_block_stop" in event_types
        assert "message_delta" in event_types
        assert "message_stop" in event_types

    async def test_stream_event_sequence_order(self, make_async_iter):
        cli_chunks = [
            {
                "type": "stream_event",
                "event": {
                    "type": "content_block_delta",
                    "delta": {"type": "text_delta", "text": "Test"},
                },
            }
        ]

        events = []
        async for event in stream_anthropic_response(
            make_async_iter(cli_chunks), "claude-sonnet-4"
        ):
            events.append(event)

        event_types = [self._extract_event_type(e) for e in events]

        # Verify order
        assert event_types.index("message_start") < event_types.index("content_block_start")
        assert event_types.index("content_block_start") < event_types.index("content_block_delta")
        assert event_types.index("content_block_delta") < event_types.index("content_block_stop")
        assert event_types.index("content_block_stop") < event_types.index("message_delta")
        assert event_types.index("message_delta") < event_types.index("message_stop")

    async def test_content_block_delta_text_extraction(self, make_async_iter):
        cli_chunks = [
            {
                "type": "stream_event",
                "event": {
                    "type": "content_block_delta",
                    "delta": {"type": "text_delta", "text": "Hello"},
                },
            },
            {
                "type": "stream_event",
                "event": {
                    "type": "content_block_delta",
                    "delta": {"type": "text_delta", "text": " world"},
                },
            },
        ]

        delta_texts = []
        async for event in stream_anthropic_response(
            make_async_iter(cli_chunks), "claude-sonnet-4"
        ):
            if "content_block_delta" in event:
                data = self._extract_event_data(event)
                if data and "delta" in data:
                    delta_texts.append(data["delta"].get("text", ""))

        assert "Hello" in delta_texts
        assert " world" in delta_texts

    async def test_legacy_streaming_format(self, make_async_iter):
        cli_chunks = [
            {
                "type": "assistant",
                "message": {
                    "content": [{"type": "text", "text": "Hello"}],
                    "usage": {"input_tokens": 10, "output_tokens": 5},
                },
            },
            {
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": " there"}]},
            },
        ]

        events = []
        async for event in stream_anthropic_response(
            make_async_iter(cli_chunks), "claude-sonnet-4"
        ):
            events.append(event)

        # Should still produce proper event sequence
        event_types = [self._extract_event_type(e) for e in events]
        assert "message_start" in event_types
        assert "content_block_delta" in event_types

    async def test_usage_tracking(self, make_async_iter):
        cli_chunks = [
            {
                "type": "stream_event",
                "event": {
                    "type": "message_delta",
                    "usage": {"output_tokens": 25},
                },
            }
        ]

        events = []
        async for event in stream_anthropic_response(
            make_async_iter(cli_chunks), "claude-sonnet-4"
        ):
            events.append(event)

        # Find message_delta event
        for event in events:
            if "message_delta" in event:
                data = self._extract_event_data(event)
                if data and "usage" in data:
                    assert data["usage"]["output_tokens"] == 25
                    break
        else:
            pytest.fail("No message_delta with usage found")

    async def test_empty_stream(self, make_async_iter):
        events = []
        async for event in stream_anthropic_response(make_async_iter([]), "claude-sonnet-4"):
            events.append(event)

        # Should still produce start and stop events
        event_types = [self._extract_event_type(e) for e in events]
        assert "message_start" in event_types
        assert "message_stop" in event_types

    async def test_message_start_event_included(self, make_async_iter):
        cli_chunks = [
            {
                "type": "stream_event",
                "event": {
                    "type": "content_block_delta",
                    "delta": {"type": "text_delta", "text": "Hi"},
                },
            }
        ]

        events = []
        async for event in stream_anthropic_response(
            make_async_iter(cli_chunks), "claude-sonnet-4"
        ):
            events.append(event)

        # Verify message_start is present
        event_types = [self._extract_event_type(e) for e in events]
        assert "message_start" in event_types

        # Verify the message_start SSE contains expected data structure
        for event in events:
            if "event: message_start" in event and "data: " in event:
                # Basic validation that SSE is well-formed
                assert event.startswith("event: message_start\n")
                assert "data: {" in event
                break

    async def test_non_dict_chunks_ignored(self, make_async_iter):
        cli_chunks = [
            "invalid string chunk",
            None,
            {
                "type": "stream_event",
                "event": {
                    "type": "content_block_delta",
                    "delta": {"type": "text_delta", "text": "Valid"},
                },
            },
        ]

        events = []
        async for event in stream_anthropic_response(
            make_async_iter(cli_chunks), "claude-sonnet-4"
        ):
            events.append(event)

        # Should still process valid chunk
        event_types = [self._extract_event_type(e) for e in events]
        assert "content_block_delta" in event_types

    # Helper methods
    def _extract_event_type(self, sse_event: str) -> str | None:
        for line in sse_event.split("\n"):
            if line.startswith("event: "):
                return line.split("event: ")[1]
        return None

    def _extract_event_data(self, sse_event: str) -> dict | None:
        for line in sse_event.split("\n"):
            if line.startswith("data: "):
                try:
                    return json.loads(line.split("data: ", 1)[1])
                except json.JSONDecodeError:
                    return None
        return None
