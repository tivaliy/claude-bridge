"""Integration tests for ClaudeClient with mocked CLI wrapper."""

from unittest.mock import patch

import pytest

from claude_bridge.core.claude_client import ClaudeClient


class TestClaudeClientQuery:
    @pytest.mark.asyncio
    async def test_query_delegates_to_cli_wrapper(self, make_async_iter):
        client = ClaudeClient()

        cli_response = [{"result": "Hello!", "usage": {"input_tokens": 5, "output_tokens": 3}}]

        with patch.object(
            client.cli, "query", return_value=make_async_iter(cli_response)
        ) as mock_query:
            results = []
            async for result in client.query("Test prompt"):
                results.append(result)

            # Verify CLI wrapper was called
            mock_query.assert_called_once()
            call_kwargs = mock_query.call_args[1]
            assert call_kwargs["prompt"] == "Test prompt"
            assert results == cli_response

    @pytest.mark.asyncio
    async def test_query_with_system_prompt(self, make_async_iter):
        client = ClaudeClient()

        with patch.object(
            client.cli, "query", return_value=make_async_iter([{"result": "Response"}])
        ) as mock_query:
            async for _ in client.query("Test", system_prompt="You are helpful"):
                pass

            call_kwargs = mock_query.call_args[1]
            assert call_kwargs["system_prompt"] == "You are helpful"

    @pytest.mark.asyncio
    async def test_query_with_model_mapping(self, make_async_iter):
        client = ClaudeClient()

        with patch.object(
            client.cli, "query", return_value=make_async_iter([{"result": "Response"}])
        ) as mock_query:
            async for _ in client.query("Test", model="claude-sonnet-4"):
                pass

            call_kwargs = mock_query.call_args[1]
            # Should be mapped to "sonnet"
            assert call_kwargs["model"] == "sonnet"

    @pytest.mark.asyncio
    async def test_query_streaming_mode(self, make_async_iter):
        client = ClaudeClient()

        with patch.object(
            client.cli, "query", return_value=make_async_iter([{"result": "Response"}])
        ) as mock_query:
            async for _ in client.query("Test", stream=True):
                pass

            call_kwargs = mock_query.call_args[1]
            assert call_kwargs["stream"] is True

    @pytest.mark.asyncio
    async def test_query_max_tokens_ignored(self, make_async_iter):
        client = ClaudeClient()

        with patch.object(
            client.cli, "query", return_value=make_async_iter([{"result": "Response"}])
        ) as mock_query:
            # max_tokens should be accepted for API compatibility
            async for _ in client.query("Test", max_tokens=2048):
                pass

            # But should not be passed to CLI
            call_kwargs = mock_query.call_args[1]
            assert "max_tokens" not in call_kwargs

    @pytest.mark.asyncio
    async def test_query_with_settings_tools(self, make_async_iter):
        with patch("claude_bridge.core.claude_client.settings") as mock_settings:
            mock_settings.claude_cli_path = "claude"
            mock_settings.claude_cwd = None
            mock_settings.claude_allowed_tools = ["Read", "Write"]
            mock_settings.claude_disallowed_tools = ["Bash"]

            client = ClaudeClient()

            with patch.object(
                client.cli, "query", return_value=make_async_iter([{"result": "Response"}])
            ) as mock_query:
                async for _ in client.query("Test"):
                    pass

                call_kwargs = mock_query.call_args[1]
                assert call_kwargs["allowed_tools"] == ["Read", "Write"]
                assert call_kwargs["disallowed_tools"] == ["Bash"]


class TestClaudeClientHealthChecks:
    @pytest.mark.asyncio
    async def test_check_available_delegates(self):
        client = ClaudeClient()

        with patch.object(client.cli, "check_available", return_value=True) as mock_check:
            result = await client.check_available()

            mock_check.assert_called_once()
            assert result is True

    @pytest.mark.asyncio
    async def test_check_available_returns_false(self):
        client = ClaudeClient()

        with patch.object(client.cli, "check_available", return_value=False):
            result = await client.check_available()

            assert result is False

    @pytest.mark.asyncio
    async def test_get_version_delegates(self):
        client = ClaudeClient()

        with patch.object(client.cli, "get_version", return_value="claude 1.0.0") as mock_version:
            version = await client.get_version()

            mock_version.assert_called_once()
            assert version == "claude 1.0.0"

    @pytest.mark.asyncio
    async def test_get_version_returns_none(self):
        client = ClaudeClient()

        with patch.object(client.cli, "get_version", return_value=None):
            version = await client.get_version()

            assert version is None


class TestClaudeClientInitialization:
    def test_initialization_uses_settings(self):
        with patch("claude_bridge.core.claude_client.settings") as mock_settings:
            mock_settings.claude_cli_path = "/custom/path/claude"
            mock_settings.claude_cwd = "/custom/cwd"

            client = ClaudeClient()

            assert client.cli.cli_path == "/custom/path/claude"
            assert client.cli.cwd == "/custom/cwd"

    def test_initialization_default_settings(self):
        client = ClaudeClient()

        # Should use default CLI path
        assert client.cli.cli_path == "claude"
        assert client.cli.cwd is None
