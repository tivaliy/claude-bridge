"""CLI entry point for Claude Bridge."""

import argparse
import sys
from pathlib import Path

import uvicorn

from . import __version__
from .app import create_app
from .config import settings


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        prog="claude-bridge",
        description="API gateway for Claude Code with Anthropic API compatibility",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Start with defaults from .env
  %(prog)s --host 127.0.0.1 --port 3000      # Custom host and port
  %(prog)s -vvv                               # Trace level logging (maximum verbosity)
  %(prog)s -p 9000 -vv                        # Port 9000 with debug logging
  %(prog)s --claude-cwd /path/to/project      # Set working directory
  %(prog)s --allowed-tools Read --allowed-directories /tmp  # Enable file upload via CLI args

Environment variables can be set via .env file or exported.
CLI arguments take precedence over environment variables.
        """,
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    # Server settings
    server_group = parser.add_argument_group("server options")
    server_group.add_argument(
        "--host",
        "-H",
        type=str,
        help=f"Server host address (default: {settings.host})",
    )
    server_group.add_argument(
        "--port",
        "-p",
        type=int,
        help=f"Server port (default: {settings.port})",
    )

    # Logging settings
    log_group = parser.add_argument_group("logging options")
    log_group.add_argument(
        "--verbose",
        "-v",
        action="count",
        default=0,
        help="Increase verbosity: -v=DEBUG, -vv=TRACE (default: INFO)",
    )

    # Claude CLI settings
    claude_group = parser.add_argument_group("Claude CLI options")
    claude_group.add_argument(
        "--claude-cli-path",
        type=str,
        help=f"Path to claude CLI binary (default: {settings.claude_cli_path})",
    )
    claude_group.add_argument(
        "--claude-cwd",
        type=str,
        help="Working directory for Claude Code operations",
    )

    # Claude CLI permissions
    perm_group = parser.add_argument_group("Claude CLI permissions (override .env settings)")
    perm_group.add_argument(
        "--allowed-tools",
        type=str,
        help="Comma-separated list of allowed tools (e.g., 'Read' or 'Read,Bash'). Overrides CLAUDE_ALLOWED_TOOLS_STR",
    )
    perm_group.add_argument(
        "--disallowed-tools",
        type=str,
        help="Comma-separated list of disallowed tools. Overrides CLAUDE_DISALLOWED_TOOLS_STR",
    )
    perm_group.add_argument(
        "--allowed-directories",
        type=str,
        help="Comma-separated list of allowed directories (absolute paths). Overrides CLAUDE_ALLOWED_DIRECTORIES_STR",
    )

    return parser.parse_args()


def main():
    """Run the Claude Bridge server."""
    try:
        args = parse_args()

        # Override settings with CLI arguments
        if args.host is not None:
            settings.host = args.host
        if args.port is not None:
            settings.port = args.port

        # Map verbosity count to log levels
        # 0 = info (default), 1 = debug, 2+ = trace
        if args.verbose >= 2:
            log_level = "trace"
            settings.debug = True
        elif args.verbose == 1:
            log_level = "debug"
            settings.debug = True
        else:
            # Use configured log level from settings (default: info)
            log_level = settings.log_level

        # Override Claude CLI settings
        if args.claude_cli_path:
            settings.claude_cli_path = args.claude_cli_path
        if args.claude_cwd:
            settings.claude_cwd = Path(args.claude_cwd)

        # Override Claude CLI permission settings (CLI args take precedence)
        if args.allowed_tools is not None:
            settings.claude_allowed_tools_str = args.allowed_tools
        if args.disallowed_tools is not None:
            settings.claude_disallowed_tools_str = args.disallowed_tools
        if args.allowed_directories is not None:
            settings.claude_allowed_directories_str = args.allowed_directories

        # Update application log level setting
        settings.log_level = log_level if log_level != "trace" else "debug"

        # Create and run app
        app = create_app()
        uvicorn.run(
            app,
            host=settings.host,
            port=settings.port,
            log_level=log_level,
        )
    except KeyboardInterrupt:
        print("\nShutting down Claude Bridge...")
        sys.exit(0)
    except Exception as e:
        print(f"Error starting Claude Bridge: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
