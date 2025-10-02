"""Integration tests for FastAPI router with mocked ClaudeClient."""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from claude_bridge.app import create_app


@pytest.fixture
def test_app():
    return create_app()


@pytest.fixture
def client(test_app):
    return TestClient(test_app)


class TestMessagesEndpointNonStreaming:
    def test_successful_request(self, client, make_async_iter):
        cli_response = [
            {
                "result": "Hello! How can I help?",
                "usage": {"input_tokens": 10, "output_tokens": 15},
            }
        ]

        with patch("claude_bridge.anthropic.router.ClaudeClient") as mock_client_class:
            mock_client = mock_client_class.return_value

            # query should be an async method that returns an async iterator
            async def mock_query(*args, **kwargs):
                for item in cli_response:
                    yield item

            mock_client.query = mock_query

            response = client.post(
                "/anthropic/v1/messages",
                json={
                    "model": "claude-sonnet-4",
                    "max_tokens": 1024,
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert data["type"] == "message"
            assert data["role"] == "assistant"
            assert len(data["content"]) == 1
            assert data["content"][0]["text"] == "Hello! How can I help?"
            assert data["model"] == "claude-sonnet-4"
            assert data["usage"]["input_tokens"] == 10
            assert data["usage"]["output_tokens"] == 15

    def test_request_with_system_prompt(self, client, make_async_iter):
        cli_response = [{"result": "4", "usage": {"input_tokens": 20, "output_tokens": 1}}]

        with patch("claude_bridge.anthropic.router.ClaudeClient") as mock_client_class:
            mock_client = mock_client_class.return_value

            async def mock_query(*args, **kwargs):
                for item in cli_response:
                    yield item

            mock_client.query = mock_query

            response = client.post(
                "/anthropic/v1/messages",
                json={
                    "model": "claude-sonnet-4",
                    "max_tokens": 1024,
                    "system": "You are a helpful assistant.",
                    "messages": [{"role": "user", "content": "What is 2+2?"}],
                },
            )

            assert response.status_code == 200

    def test_missing_user_message_returns_400(self, client):
        response = client.post(
            "/anthropic/v1/messages",
            json={
                "model": "claude-sonnet-4",
                "max_tokens": 1024,
                "messages": [{"role": "assistant", "content": "Hello"}],
            },
        )

        assert response.status_code == 400
        assert "No user message found" in response.json()["detail"]

    def test_empty_messages_list_returns_400(self, client):
        response = client.post(
            "/anthropic/v1/messages",
            json={
                "model": "claude-sonnet-4",
                "max_tokens": 1024,
                "messages": [],
            },
        )

        assert response.status_code == 400

    def test_invalid_request_schema_returns_422(self, client):
        response = client.post(
            "/anthropic/v1/messages",
            json={
                "model": "claude-sonnet-4",
                # Missing required 'messages' field
                "max_tokens": 1024,
            },
        )

        assert response.status_code == 422

    def test_cli_error_returns_500(self, client, make_async_iter):
        with patch("claude_bridge.anthropic.router.ClaudeClient") as mock_client_class:
            mock_client = mock_client_class.return_value

            async def mock_query(*args, **kwargs):
                raise RuntimeError("CLI failed")
                yield  # Make it a generator

            mock_client.query = mock_query

            response = client.post(
                "/anthropic/v1/messages",
                json={
                    "model": "claude-sonnet-4",
                    "max_tokens": 1024,
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            )

            assert response.status_code == 500
            assert "CLI failed" in response.json()["detail"]

    def test_multi_turn_conversation(self, client, make_async_iter):
        cli_response = [
            {
                "result": "Your name is Bob!",
                "usage": {"input_tokens": 50, "output_tokens": 10},
            }
        ]

        with patch("claude_bridge.anthropic.router.ClaudeClient") as mock_client_class:
            mock_client = mock_client_class.return_value

            # Capture the prompt that was sent
            captured_prompt = None

            async def mock_query(*args, prompt=None, **kwargs):
                nonlocal captured_prompt
                captured_prompt = prompt
                for item in cli_response:
                    yield item

            mock_client.query = mock_query

            response = client.post(
                "/anthropic/v1/messages",
                json={
                    "model": "claude-sonnet-4",
                    "max_tokens": 1024,
                    "messages": [
                        {"role": "user", "content": "My name is Bob"},
                        {"role": "assistant", "content": "Hello Bob!"},
                        {"role": "user", "content": "What's my name?"},
                    ],
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert data["content"][0]["text"] == "Your name is Bob!"

            # Verify the prompt includes conversation context in XML format
            assert captured_prompt is not None
            assert "<conversation>" in captured_prompt
            assert '<turn role="user">My name is Bob</turn>' in captured_prompt
            assert '<turn role="assistant">Hello Bob!</turn>' in captured_prompt
            assert "What's my name?" in captured_prompt


class TestMessagesEndpointStreaming:
    def test_streaming_request_returns_sse(self, client, make_async_iter):
        cli_chunks = [
            {
                "type": "stream_event",
                "event": {
                    "type": "content_block_delta",
                    "delta": {"type": "text_delta", "text": "Hello"},
                },
            }
        ]

        with patch("claude_bridge.anthropic.router.ClaudeClient") as mock_client_class:
            mock_client = mock_client_class.return_value

            async def mock_query(*args, **kwargs):
                for item in cli_chunks:
                    yield item

            mock_client.query = mock_query

            response = client.post(
                "/anthropic/v1/messages",
                json={
                    "model": "claude-sonnet-4",
                    "max_tokens": 1024,
                    "stream": True,
                    "messages": [{"role": "user", "content": "Count to 5"}],
                },
            )

            assert response.status_code == 200
            assert response.headers["content-type"] == "text/event-stream; charset=utf-8"
            assert "Cache-Control" in response.headers
            assert response.headers["Cache-Control"] == "no-cache"

    def test_streaming_content_format(self, client, make_async_iter):
        cli_chunks = [
            {
                "type": "stream_event",
                "event": {
                    "type": "content_block_delta",
                    "delta": {"type": "text_delta", "text": "Test"},
                },
            }
        ]

        with patch("claude_bridge.anthropic.router.ClaudeClient") as mock_client_class:
            mock_client = mock_client_class.return_value

            async def mock_query(*args, **kwargs):
                for item in cli_chunks:
                    yield item

            mock_client.query = mock_query

            response = client.post(
                "/anthropic/v1/messages",
                json={
                    "model": "claude-sonnet-4",
                    "max_tokens": 1024,
                    "stream": True,
                    "messages": [{"role": "user", "content": "Test"}],
                },
            )

            content = response.text
            # Should contain SSE events
            assert "event: message_start" in content
            assert "data: " in content


class TestHealthEndpoint:
    @pytest.mark.asyncio
    async def test_health_check_cli_available(self, client):
        with patch("claude_bridge.app.ClaudeClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.check_available.return_value = True
            mock_client.get_version.return_value = "claude 1.0.0"
            mock_client_class.return_value = mock_client

            response = client.get("/health")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert data["cli_available"] is True
            assert data["cli_version"] == "claude 1.0.0"

    @pytest.mark.asyncio
    async def test_health_check_cli_unavailable(self, client):
        with patch("claude_bridge.app.ClaudeClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.check_available.return_value = False
            mock_client.get_version.return_value = None
            mock_client_class.return_value = mock_client

            response = client.get("/health")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "degraded"
            assert data["cli_available"] is False
            assert data["cli_version"] is None


class TestRootEndpoint:
    def test_root_endpoint(self, client):
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert "service" in data
        assert "version" in data
        assert "providers" in data
        assert "anthropic" in data["providers"]
