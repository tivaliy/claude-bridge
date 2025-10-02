"""Configuration management for Claude Bridge."""

import logging
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings."""

    # Application settings
    app_name: str = "Claude Bridge"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8080

    # Logging / observability
    # Log level for application (debug, info, warning, error, critical)
    log_level: str = "info"
    # Enable structured JSON logging in future (currently unused placeholder)
    log_json: bool = False

    # Claude Code CLI settings
    # Path to claude CLI binary (default: "claude")
    claude_cli_path: str = "claude"

    # Optional: Set working directory for Claude Code operations
    claude_cwd: Path | None = None

    # Optional: Control which tools Claude Code can use (legacy, kept for compatibility)
    claude_allowed_tools: list[str] | None = None
    claude_disallowed_tools: list[str] | None = None

    # Claude CLI Permission Configuration
    # IMPORTANT: These settings control file access for image/PDF upload
    # Leave empty to disable file upload (secure default)

    # Comma-separated list of allowed tools (e.g., "Read" or "Read,Bash")
    # Empty string = file upload disabled
    # Example: "Read" to enable image/PDF analysis
    claude_allowed_tools_str: str = ""

    # Comma-separated list of disallowed tools (optional)
    claude_disallowed_tools_str: str = ""

    # Permission mode for non-interactive API usage
    # - bypassPermissions: Skip interactive prompts (required for API mode)
    # - default: Interactive mode (will fail in API context)
    # - acceptEdits: Auto-accept edit operations
    claude_permission_mode: str = "bypassPermissions"

    # Comma-separated list of allowed directories (absolute paths)
    # Empty string = no file access
    # Example: "/tmp" or "/tmp,/var/app-temp"
    # User MUST explicitly configure this for image/PDF upload to work
    claude_allowed_directories_str: str = ""

    # Process management / safety
    # Maximum time (seconds) to allow a single non-streaming CLI invocation to run
    claude_process_timeout_seconds: int = 300
    # Maximum idle time (seconds) without new stdout lines during streaming before aborting
    claude_stream_idle_timeout_seconds: int = 180
    # Grace period (seconds) after sending kill before force terminating
    claude_process_kill_grace_seconds: int = 5

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        # Allow ANTHROPIC_API_KEY to be read from environment
        env_prefix="",
    )

    def configure_logging(self) -> None:
        """Configure root logging based on settings.

        Sets up basic logging with the configured log level.
        Can be extended to support structured JSON logging if log_json is True.
        """
        level = getattr(logging, self.log_level.upper(), logging.INFO)

        # Avoid reconfiguring if handlers already exist (e.g., in reload/debug mode)
        if logging.getLogger().handlers:
            logging.getLogger().setLevel(level)
            return

        logging.basicConfig(
            level=level,
            format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        )


settings = Settings()
