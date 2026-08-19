"""Claude Code CLI client wrapper."""

from collections.abc import AsyncIterator
from typing import Any

from ..config import settings
from .cli_wrapper import ClaudeCLIWrapper
from .model_mapper import ModelMapper


class ClaudeClient:
    """Wrapper for Claude Code CLI providing stateless query interface."""

    def __init__(self):
        """
        Initialize Claude client using CLI wrapper.

        Note: the bridge passes no credentials; the CLI resolves its own, and
        ANTHROPIC_API_KEY takes precedence over a claude.ai/subscription login
        (an invalid key fails with 401 rather than falling back to it). Bedrock
        and Vertex credentials are also honored. See README -> Authentication for
        which of these is a supported way to run the bridge.
        """
        self.cli = ClaudeCLIWrapper(
            cli_path=settings.claude_cli_path,
            cwd=settings.claude_cwd,
        )

    async def query(
        self,
        prompt: str,
        system_prompt: str | None = None,
        model: str | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> AsyncIterator[dict[str, Any]]:
        """
        Execute a single stateless query to Claude Code CLI.

        Args:
            prompt: The user prompt/message
            system_prompt: Optional system prompt
            model: Model to use (e.g., 'sonnet', 'opus')
            stream: Whether to stream responses
            **kwargs: Additional options (max_tokens accepted but ignored for API compatibility)

        Yields:
            Parsed JSON responses from Claude CLI
        """
        # Map model name to CLI-compatible format
        cli_model = ModelMapper.map_model(model) if model else None

        # Note: max_tokens is accepted in kwargs for Anthropic API compatibility
        # but is not passed to CLI as it's not supported by --print mode
        async for response in self.cli.query(
            prompt=prompt,
            system_prompt=system_prompt,
            model=cli_model,
            allowed_tools=settings.claude_allowed_tools,
            disallowed_tools=settings.claude_disallowed_tools,
            stream=stream,
        ):
            yield response

    async def check_available(self) -> bool:
        """Check if Claude CLI is available."""
        result = await self.cli.check_available()
        return bool(result)

    async def get_version(self) -> str | None:
        """Get Claude CLI version."""
        return await self.cli.get_version()  # type: ignore[no-any-return]
