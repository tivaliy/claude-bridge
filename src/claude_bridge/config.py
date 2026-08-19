"""Configuration management for Claude Bridge."""

import logging
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Tools denied to the CLI unless the operator overrides
# CLAUDE_DISALLOWED_TOOLS_STR.
#
# Why a deny-list and not an allow-list: under --permission-mode bypassPermissions
# the CLI's --allowed-tools flag pre-approves tools but does not withhold the rest,
# so Bash and Write stay reachable however narrow the allow-list is. Only
# --disallowed-tools actually withholds a tool. That makes this list the security
# boundary, and it has to be enumerated rather than derived.
#
# What survives: Read, Glob, Grep — enough for the file-upload path, none of which
# executes, mutates, persists, or reaches the network.
#
# LIMITATION: a deny-list cannot cover tools that do not exist yet. A CLI upgrade
# that adds a tool grants it to every caller of this API until it is added here.
# To re-check on upgrade, ask the CLI what it kept:
#   echo "List every tool name you have available, one per line." \
#     | claude --print --strict-mcp-config --permission-mode bypassPermissions \
#         --disallowed-tools "<the names below, space-separated>"
# Stale names are reported too: the CLI warns on stderr per unmatched deny rule
# ("matches no known tool"), which is how the dead MultiEdit entry was found.
DEFAULT_DISALLOWED_TOOLS = ",".join(
    (
        # Execution and mutation
        "Bash",
        "Write",
        "Edit",
        "NotebookEdit",
        # Network egress
        "WebFetch",
        "WebSearch",
        # Delegation — a subagent is a fresh tool surface this list cannot reach
        "Agent",
        "Task",
        "TaskCreate",
        "TaskGet",
        "TaskList",
        "TaskOutput",
        "TaskStop",
        "TaskUpdate",
        "Workflow",
        "Skill",
        "ToolSearch",
        "ListAgents",
        # Anything that outlives the request, or speaks to someone
        "CronCreate",
        "CronDelete",
        "CronList",
        "ScheduleWakeup",
        "Monitor",
        "SendMessage",
        "RemoteTrigger",
        "PushNotification",
        "ShareOnboardingGuide",
        "DesignSync",
        "ReportFindings",
        "EnterWorktree",
        "ExitWorktree",
        # MCP plumbing — belt and braces alongside --strict-mcp-config
        "ListMcpResourcesTool",
        "ReadMcpResourceTool",
        "ReadMcpResourceDirTool",
    )
)


class Settings(BaseSettings):
    """Application settings."""

    # Application settings
    app_name: str = "Claude Bridge"
    debug: bool = False
    # Loopback by default: the bridge has no authentication of its own, so anything
    # that can reach the port can spend your credentials.
    host: str = "127.0.0.1"

    # Comma-separated CORS origins. Empty means no cross-origin access, so a page in
    # the operator's browser cannot drive a local instance.
    cors_allow_origins_str: str = ""
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

    # Comma-separated. Treat every prompt reaching this API as untrusted input —
    # instructions hidden in an uploaded document are indistinguishable from the
    # caller's own. See DEFAULT_DISALLOWED_TOOLS above for why this list, and not
    # the allow-list, is the security boundary. Setting this replaces the default;
    # "" lifts the restriction entirely.
    claude_disallowed_tools_str: str = DEFAULT_DISALLOWED_TOOLS

    # Permission mode for non-interactive API usage.
    # NOTE: not currently honored — cli_wrapper always passes bypassPermissions,
    # which is the only mode an HTTP caller can work under (it cannot answer a prompt).
    # - bypassPermissions: Skip interactive prompts (required for API mode)
    # - default: Interactive mode (will fail in API context)
    # - acceptEdits: Auto-accept edit operations
    claude_permission_mode: str = "bypassPermissions"

    # Comma-separated list of allowed directories (absolute paths)
    # Empty string = no file access
    # Example: "/tmp" or "/tmp,/var/app-temp"
    # User MUST explicitly configure this for image/PDF upload to work
    claude_allowed_directories_str: str = ""

    # Load no MCP servers unless they are passed explicitly. Without this the CLI
    # inherits the operator's personal MCP configuration, so an HTTP caller reaches
    # whatever those servers expose — measured: a browser-automation server was
    # callable through this API on a stock developer machine.
    claude_strict_mcp_config: bool = True

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
