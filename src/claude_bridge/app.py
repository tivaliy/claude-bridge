"""FastAPI application factory for Claude Bridge."""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import __version__
from .anthropic import router as anthropic_router
from .config import settings
from .core.claude_client import ClaudeClient


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    settings.configure_logging()
    logger = logging.getLogger("claude_bridge.app")
    logger.debug("Initializing FastAPI application")

    app = FastAPI(
        title=settings.app_name,
        description="API gateway for Claude Code with Anthropic API compatibility",
        version=__version__,
        debug=settings.debug,
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
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
