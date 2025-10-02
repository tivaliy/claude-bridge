"""Shared test fixtures for Claude Bridge tests."""

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest

# Load test fixtures
FIXTURES_DIR = Path(__file__).parent / "fixtures"


def load_fixture(filename: str) -> dict[str, Any]:
    """Load JSON fixture file."""
    with open(FIXTURES_DIR / filename) as f:
        return json.load(f)


# Test data fixtures
@pytest.fixture
def cli_responses() -> dict[str, Any]:
    """Load CLI response fixtures."""
    return load_fixture("cli_responses.json")


@pytest.fixture
def anthropic_requests() -> dict[str, Any]:
    """Load Anthropic request fixtures."""
    return load_fixture("anthropic_requests.json")


# Mock subprocess fixtures
@pytest.fixture
def mock_subprocess_process():
    """Create a mock subprocess.Process object."""

    def _create_process(
        returncode: int = 0,
        stdout: str = "",
        stderr: str = "",
    ) -> Mock:
        """Create mock process with configurable output."""
        process = Mock()
        process.returncode = returncode
        process.stdout = Mock()
        process.stderr = Mock()
        # Mock stdin for writing
        stdin_mock = Mock()
        stdin_mock.write = Mock()
        stdin_mock.drain = AsyncMock(return_value=None)
        stdin_mock.close = Mock()
        process.stdin = stdin_mock

        # Mock readline for streaming
        stdout_lines = stdout.encode().split(b"\n") if stdout else []

        async def mock_readline():
            if stdout_lines:
                line = stdout_lines.pop(0)
                return line + b"\n" if line else b""
            return b""

        process.stdout.readline = AsyncMock(side_effect=mock_readline)

        # Mock communicate for non-streaming
        async def mock_communicate(input=None):
            return stdout.encode(), stderr.encode()

        process.communicate = AsyncMock(side_effect=mock_communicate)

        # Mock wait
        async def mock_wait():
            return returncode

        process.wait = AsyncMock(side_effect=mock_wait)

        # Mock terminate and kill
        process.terminate = Mock()
        process.kill = Mock()

        return process

    return _create_process


@pytest.fixture
def mock_create_subprocess_exec(mock_subprocess_process):
    """Mock asyncio.create_subprocess_exec."""

    def _mock_exec(returncode: int = 0, stdout: str = "", stderr: str = ""):
        """Create mock that returns process with given outputs."""
        process = mock_subprocess_process(returncode, stdout, stderr)

        async def mock_create(*args, **kwargs):
            return process

        return AsyncMock(side_effect=mock_create)

    return _mock_exec


# Async iterator helpers
async def async_iter_list(items: list[Any]):
    """Convert list to async iterator."""
    for item in items:
        yield item


@pytest.fixture
def make_async_iter():
    """Factory to create async iterators from lists."""
    return async_iter_list


# CLI response generators
@pytest.fixture
def cli_success_response(cli_responses):
    """CLI success response."""
    return cli_responses["success_response"]


@pytest.fixture
def cli_error_response(cli_responses):
    """CLI error response."""
    return cli_responses["error_response"]


@pytest.fixture
def cli_streaming_chunks(cli_responses):
    """CLI streaming response chunks."""
    return cli_responses["streaming_chunks"]


@pytest.fixture
def cli_legacy_streaming_chunks(cli_responses):
    """CLI legacy streaming response chunks."""
    return cli_responses["legacy_streaming_chunks"]


# Anthropic request generators
@pytest.fixture
def simple_anthropic_request(anthropic_requests):
    """Simple Anthropic API request."""
    return anthropic_requests["simple_request"]


@pytest.fixture
def anthropic_request_with_system(anthropic_requests):
    """Anthropic request with system prompt."""
    return anthropic_requests["request_with_system"]


@pytest.fixture
def streaming_anthropic_request(anthropic_requests):
    """Anthropic streaming request."""
    return anthropic_requests["streaming_request"]


@pytest.fixture
def multi_message_request(anthropic_requests):
    """Multi-message conversation request."""
    return anthropic_requests["multi_message_request"]


@pytest.fixture
def complex_content_request(anthropic_requests):
    """Request with complex content blocks."""
    return anthropic_requests["complex_content_request"]
