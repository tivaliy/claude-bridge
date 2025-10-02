"""Claude Bridge - API gateway for Claude Code with Anthropic API compatibility."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("claude-bridge")
except PackageNotFoundError:
    # Package is not installed, use fallback version
    __version__ = "0.0.0+dev"

__all__ = ["__version__"]
