"""Main entry point for Claude Bridge.

Note: The recommended way to start the server is using the CLI command:
    claude-bridge

Or with uv:
    uv run claude-bridge

This file can still be used for direct execution:
    python main.py
"""

import uvicorn

from src.claude_bridge.app import create_app
from src.claude_bridge.config import settings


def main():
    """Run the FastAPI application."""
    app = create_app()
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        log_level="debug" if settings.debug else "info",
    )


if __name__ == "__main__":
    main()
