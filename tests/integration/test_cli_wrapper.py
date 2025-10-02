"""Integration tests for CLI wrapper with mocked subprocess."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from claude_bridge.core.cli_wrapper import ClaudeCLIWrapper


class TestCLIWrapperCommandBuilding:
    @pytest.mark.asyncio
    async def test_basic_command_construction(self, mock_subprocess_process):
        cli = ClaudeCLIWrapper()
        process = mock_subprocess_process(returncode=0, stdout=json.dumps({"result": "test"}))

        with patch("asyncio.create_subprocess_exec", return_value=process) as mock_exec:
            async for _ in cli.query("Hello"):
                pass

            # Verify command structure
            call_args = mock_exec.call_args
            cmd = call_args[0]
            assert cmd[0] == "claude"
            assert "--print" in cmd
            assert "--output-format" in cmd
            assert "json" in cmd
            assert "--permission-mode" in cmd
            assert "bypassPermissions" in cmd

    @pytest.mark.asyncio
    async def test_streaming_mode_flags(self, mock_subprocess_process):
        cli = ClaudeCLIWrapper()
        process = mock_subprocess_process(returncode=0)

        with patch("asyncio.create_subprocess_exec", return_value=process) as mock_exec:
            async for _ in cli.query("Test", stream=True):
                pass

            cmd = mock_exec.call_args[0]
            assert "--verbose" in cmd
            assert "--output-format" in cmd
            assert "stream-json" in cmd
            assert "--include-partial-messages" in cmd

    @pytest.mark.asyncio
    async def test_model_argument(self, mock_subprocess_process):
        cli = ClaudeCLIWrapper()
        process = mock_subprocess_process(returncode=0, stdout=json.dumps({"result": "test"}))

        with patch("asyncio.create_subprocess_exec", return_value=process) as mock_exec:
            async for _ in cli.query("Hello", model="sonnet"):
                pass

            cmd = mock_exec.call_args[0]
            assert "--model" in cmd
            assert "sonnet" in cmd

    @pytest.mark.asyncio
    async def test_system_prompt_argument(self, mock_subprocess_process):
        cli = ClaudeCLIWrapper()
        process = mock_subprocess_process(returncode=0, stdout=json.dumps({"result": "test"}))

        with patch("asyncio.create_subprocess_exec", return_value=process) as mock_exec:
            async for _ in cli.query("Hello", system_prompt="You are helpful"):
                pass

            cmd = mock_exec.call_args[0]
            assert "--append-system-prompt" in cmd
            assert "You are helpful" in cmd

    @pytest.mark.asyncio
    async def test_allowed_tools_argument(self, mock_subprocess_process):
        cli = ClaudeCLIWrapper()
        process = mock_subprocess_process(returncode=0, stdout=json.dumps({"result": "test"}))

        with patch("asyncio.create_subprocess_exec", return_value=process) as mock_exec:
            async for _ in cli.query("Hello", allowed_tools=["Read", "Write"]):
                pass

            cmd = mock_exec.call_args[0]
            assert "--allowed-tools" in cmd
            # Tools should be space-separated
            tools_idx = cmd.index("--allowed-tools") + 1
            assert "Read Write" in cmd[tools_idx]

    @pytest.mark.asyncio
    async def test_disallowed_tools_argument(self, mock_subprocess_process):
        cli = ClaudeCLIWrapper()
        process = mock_subprocess_process(returncode=0, stdout=json.dumps({"result": "test"}))

        with patch("asyncio.create_subprocess_exec", return_value=process) as mock_exec:
            async for _ in cli.query("Hello", disallowed_tools=["Bash"]):
                pass

            cmd = mock_exec.call_args[0]
            assert "--disallowed-tools" in cmd

    @pytest.mark.asyncio
    async def test_cwd_argument(self, mock_subprocess_process):
        cli = ClaudeCLIWrapper(cwd=Path("/tmp/test"))
        process = mock_subprocess_process(returncode=0, stdout=json.dumps({"result": "test"}))

        with patch("asyncio.create_subprocess_exec", return_value=process) as mock_exec:
            async for _ in cli.query("Hello"):
                pass

            kwargs = mock_exec.call_args[1]
            assert kwargs["cwd"] == "/tmp/test"


class TestCLIWrapperNonStreamingResponse:
    @pytest.mark.asyncio
    async def test_successful_json_response(self, mock_subprocess_process):
        cli = ClaudeCLIWrapper()
        response_data = {
            "result": "Hello! How can I help?",
            "usage": {"input_tokens": 10, "output_tokens": 15},
        }
        process = mock_subprocess_process(returncode=0, stdout=json.dumps(response_data))

        with patch("asyncio.create_subprocess_exec", return_value=process):
            results = []
            async for result in cli.query("Hello"):
                results.append(result)

            assert len(results) == 1
            assert results[0] == response_data

    @pytest.mark.asyncio
    async def test_non_zero_exit_code_raises(self, mock_subprocess_process):
        cli = ClaudeCLIWrapper()
        process = mock_subprocess_process(returncode=1, stderr="Error occurred")

        with patch("asyncio.create_subprocess_exec", return_value=process):
            with pytest.raises(RuntimeError, match="Claude CLI failed"):
                async for _ in cli.query("Hello"):
                    pass

    @pytest.mark.asyncio
    async def test_is_error_flag_raises(self, mock_subprocess_process):
        cli = ClaudeCLIWrapper()
        error_response = {"is_error": True, "result": "Model not found"}
        process = mock_subprocess_process(returncode=0, stdout=json.dumps(error_response))

        with patch("asyncio.create_subprocess_exec", return_value=process):
            with pytest.raises(RuntimeError, match="Model not found"):
                async for _ in cli.query("Hello"):
                    pass

    @pytest.mark.asyncio
    async def test_unsupported_model_error_in_stderr(self, mock_subprocess_process):
        cli = ClaudeCLIWrapper()
        # Simulate actual CLI error for unsupported model
        stderr_msg = "Error: Unknown model 'gpt-4-turbo'"
        process = mock_subprocess_process(returncode=1, stderr=stderr_msg)

        with patch("asyncio.create_subprocess_exec", return_value=process):
            with pytest.raises(RuntimeError, match=r"Claude CLI failed \(exit 1\):.*gpt-4-turbo"):
                async for _ in cli.query("Hello", model="gpt-4-turbo"):
                    pass

    @pytest.mark.asyncio
    async def test_unsupported_model_error_with_empty_stderr(self, mock_subprocess_process):
        cli = ClaudeCLIWrapper()
        # Simulate the case where stderr is empty but CLI fails
        process = mock_subprocess_process(returncode=1, stderr="")

        with patch("asyncio.create_subprocess_exec", return_value=process):
            with pytest.raises(
                RuntimeError, match=r"Claude CLI failed \(exit 1\): Command failed with exit code 1"
            ):
                async for _ in cli.query("Hello", model="gpt-4"):
                    pass

    @pytest.mark.asyncio
    async def test_model_error_in_json_response(self, mock_subprocess_process):
        cli = ClaudeCLIWrapper()
        # Some errors might be returned in JSON format
        error_response = {
            "is_error": True,
            "result": "The model 'gemini-pro' is not supported by Claude CLI",
        }
        process = mock_subprocess_process(returncode=0, stdout=json.dumps(error_response))

        with patch("asyncio.create_subprocess_exec", return_value=process):
            with pytest.raises(RuntimeError, match="gemini-pro.*not supported"):
                async for _ in cli.query("Hello", model="gemini-pro"):
                    pass

    @pytest.mark.asyncio
    async def test_api_error_in_json_with_exit_code_1(self, mock_subprocess_process):
        cli = ClaudeCLIWrapper()
        # Simulate real CLI behavior: exit code 1 with error in JSON stdout
        error_response = {
            "type": "result",
            "is_error": True,
            "result": 'API Error: 400 {"type":"error","error":{"type":"invalid_request_error","message":"Could not process image"}}',
        }
        process = mock_subprocess_process(returncode=1, stdout=json.dumps(error_response))

        with patch("asyncio.create_subprocess_exec", return_value=process):
            with pytest.raises(
                RuntimeError, match=r"Claude CLI failed \(exit 1\):.*Could not process image"
            ):
                async for _ in cli.query("Hello"):
                    pass

    @pytest.mark.asyncio
    async def test_invalid_json_raises(self, mock_subprocess_process):
        cli = ClaudeCLIWrapper()
        process = mock_subprocess_process(returncode=0, stdout="Not valid JSON")

        with patch("asyncio.create_subprocess_exec", return_value=process):
            with pytest.raises(RuntimeError, match="Failed to parse CLI output"):
                async for _ in cli.query("Hello"):
                    pass

    @pytest.mark.asyncio
    async def test_empty_stdout(self, mock_subprocess_process):
        cli = ClaudeCLIWrapper()
        process = mock_subprocess_process(returncode=0, stdout="")

        with patch("asyncio.create_subprocess_exec", return_value=process):
            results = []
            async for result in cli.query("Hello"):
                results.append(result)

            # Should yield nothing for empty stdout
            assert len(results) == 0


class TestCLIWrapperStreamingResponse:
    @pytest.mark.asyncio
    async def test_streaming_json_parsing(self):
        cli = ClaudeCLIWrapper()

        # Create process with multiple JSON lines
        chunks = [
            {"type": "content", "text": "Hello"},
            {"type": "content", "text": " world"},
            {"type": "usage", "usage": {"input_tokens": 5, "output_tokens": 2}},
        ]
        stdout = "\n".join(json.dumps(chunk) for chunk in chunks)

        # Mock process
        process = Mock()
        process.returncode = 0
        process.stdout = Mock()

        # Mock stdin for streaming
        stdin_mock = Mock()
        stdin_mock.write = Mock()
        stdin_mock.drain = AsyncMock(return_value=None)
        stdin_mock.close = Mock()
        process.stdin = stdin_mock

        # Create async readline that yields lines
        lines = [line.encode() + b"\n" for line in stdout.split("\n")] + [b""]
        line_iter = iter(lines)

        async def mock_readline():
            try:
                return next(line_iter)
            except StopIteration:
                return b""

        process.stdout.readline = mock_readline

        async def mock_wait():
            return 0

        process.wait = mock_wait
        process.terminate = Mock()

        with patch("asyncio.create_subprocess_exec", return_value=process):
            results = []
            async for result in cli.query("Test", stream=True):
                results.append(result)

            assert len(results) == 3
            assert results[0] == chunks[0]
            assert results[1] == chunks[1]
            assert results[2] == chunks[2]

    @pytest.mark.asyncio
    async def test_streaming_skips_invalid_json(self):
        cli = ClaudeCLIWrapper()

        stdout = """{"type": "valid"}
invalid line
{"type": "also valid"}"""

        # Mock process
        process = Mock()
        process.returncode = 0
        process.stdout = Mock()

        # Mock stdin for streaming
        stdin_mock = Mock()
        stdin_mock.write = Mock()
        stdin_mock.drain = AsyncMock(return_value=None)
        stdin_mock.close = Mock()
        process.stdin = stdin_mock

        lines = [line.encode() + b"\n" for line in stdout.split("\n")] + [b""]
        line_iter = iter(lines)

        async def mock_readline():
            try:
                return next(line_iter)
            except StopIteration:
                return b""

        process.stdout.readline = mock_readline

        async def mock_wait():
            return 0

        process.wait = mock_wait
        process.terminate = Mock()

        with patch("asyncio.create_subprocess_exec", return_value=process):
            results = []
            async for result in cli.query("Test", stream=True):
                results.append(result)

            # Should skip invalid line
            assert len(results) == 2
            assert results[0]["type"] == "valid"
            assert results[1]["type"] == "also valid"


class TestCLIWrapperFileUploadValidation:
    @pytest.mark.asyncio
    async def test_file_upload_without_tools_configured(self, mock_subprocess_process, monkeypatch):
        from pathlib import Path

        from claude_bridge.config import settings

        # Clear tools configuration
        monkeypatch.setattr(settings, "claude_allowed_tools_str", "")
        monkeypatch.setattr(settings, "claude_allowed_directories_str", "/tmp")

        cli = ClaudeCLIWrapper()
        process = mock_subprocess_process(returncode=0, stdout='{"result": "test"}')

        with patch("asyncio.create_subprocess_exec", return_value=process):
            with pytest.raises(ValueError, match="CLAUDE_ALLOWED_TOOLS_STR not configured"):
                async for _ in cli.query("Test", file_paths=[Path("/tmp/test.png")]):
                    pass

    @pytest.mark.asyncio
    async def test_file_upload_without_directories_configured(
        self, mock_subprocess_process, monkeypatch
    ):
        """Test that file upload raises error when CLAUDE_ALLOWED_DIRECTORIES_STR not configured."""
        from pathlib import Path

        from claude_bridge.config import settings

        # Clear directories configuration
        monkeypatch.setattr(settings, "claude_allowed_tools_str", "Read")
        monkeypatch.setattr(settings, "claude_allowed_directories_str", "")

        cli = ClaudeCLIWrapper()
        process = mock_subprocess_process(returncode=0, stdout='{"result": "test"}')

        with patch("asyncio.create_subprocess_exec", return_value=process):
            with pytest.raises(ValueError, match="CLAUDE_ALLOWED_DIRECTORIES_STR not configured"):
                async for _ in cli.query("Test", file_paths=[Path("/tmp/test.png")]):
                    pass


class TestCLIWrapperHealthChecks:
    @pytest.mark.asyncio
    async def test_check_available_returns_true(self, mock_subprocess_process):
        cli = ClaudeCLIWrapper()
        process = mock_subprocess_process(returncode=0)

        with patch("asyncio.create_subprocess_exec", return_value=process):
            result = await cli.check_available()
            assert result is True

    @pytest.mark.asyncio
    async def test_check_available_returns_false_on_error(self, mock_subprocess_process):
        cli = ClaudeCLIWrapper()
        process = mock_subprocess_process(returncode=1)

        with patch("asyncio.create_subprocess_exec", return_value=process):
            result = await cli.check_available()
            assert result is False

    @pytest.mark.asyncio
    async def test_check_available_returns_false_on_file_not_found(self):
        cli = ClaudeCLIWrapper()

        with patch("asyncio.create_subprocess_exec", side_effect=FileNotFoundError):
            result = await cli.check_available()
            assert result is False

    @pytest.mark.asyncio
    async def test_get_version_returns_version_string(self, mock_subprocess_process):
        cli = ClaudeCLIWrapper()
        process = mock_subprocess_process(returncode=0, stdout="claude 1.0.0")

        with patch("asyncio.create_subprocess_exec", return_value=process):
            version = await cli.get_version()
            assert version == "claude 1.0.0"

    @pytest.mark.asyncio
    async def test_get_version_returns_none_on_error(self, mock_subprocess_process):
        cli = ClaudeCLIWrapper()
        process = mock_subprocess_process(returncode=1)

        with patch("asyncio.create_subprocess_exec", return_value=process):
            version = await cli.get_version()
            assert version is None

    @pytest.mark.asyncio
    async def test_get_version_returns_none_on_file_not_found(self):
        cli = ClaudeCLIWrapper()

        with patch("asyncio.create_subprocess_exec", side_effect=FileNotFoundError):
            version = await cli.get_version()
            assert version is None
