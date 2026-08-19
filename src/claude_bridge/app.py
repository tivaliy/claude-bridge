"""FastAPI application factory for Claude Bridge."""

import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import __version__
from .anthropic import router as anthropic_router
from .config import settings
from .core.claude_client import ClaudeClient


def _warn_if_no_api_key(logger: logging.Logger) -> None:
    """Warn when the CLI will fall back to subscription OAuth.

    The bridge holds no credentials of its own; the CLI authenticates with either an
    API key or a Pro/Max subscription token. Anthropic reserves the latter for ordinary
    interactive use of Claude Code, not for fronting an HTTP API, so an operator who
    reaches this state should know they are in it rather than discover it later.
    """
    if os.environ.get("ANTHROPIC_API_KEY"):
        return
    logger.warning(
        "No ANTHROPIC_API_KEY in the environment: the Claude CLI will authenticate with "
        "whatever it has, which is usually a Pro/Max subscription token. Anthropic's terms "
        "reserve that credential for ordinary interactive use of Claude Code and direct "
        "developers to API keys instead, so do not serve other users from this instance. "
        "See https://code.claude.com/docs/en/legal-and-compliance"
    )


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    settings.configure_logging()
    logger = logging.getLogger("claude_bridge.app")
    logger.debug("Initializing FastAPI application")
    _warn_if_no_api_key(logger)

    app = FastAPI(
        title=settings.app_name,
        description="API gateway for Claude Code with Anthropic API compatibility",
        version=__version__,
        debug=settings.debug,
    )

    # Add CORS middleware
    # allow_credentials is off: paired with a wildcard origin Starlette ignores the
    # wildcard anyway, and the combination is what would let a browser page drive
    # a local instance with the caller's cookies.
    cors_origins = [o.strip() for o in settings.cors_allow_origins_str.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=False,
        allow_methods=["POST", "GET", "OPTIONS"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(anthropic_router)

    # Root endpoint
    @app.get("/")
    async def root():
        return {
            "service": settings.app_name,
            "version": __version__,
            "providers": ["anthropic"],
        }

    # Global health check
    @app.get("/health")
    async def health():
        claude_client = ClaudeClient()
        cli_available = await claude_client.check_available()
        cli_version = await claude_client.get_version()

        logger.debug(
            "Health check", extra={"cli_available": cli_available, "cli_version": cli_version}
        )

        return {
            "status": "healthy" if cli_available else "degraded",
            "service": settings.app_name,
            "cli_available": cli_available,
            "cli_version": cli_version,
        }

    logger.debug("Application initialization complete")
    return app
