"""Unit tests for configuration management."""

import logging
from pathlib import Path
from unittest.mock import patch

from claude_bridge.config import Settings


class TestSettings:
    def test_default_values(self):
        # Use empty env dict to ensure clean state
        # Also need to prevent .env file loading
        with patch.dict("os.environ", {}, clear=True):
            settings = Settings(_env_file=None)  # Disable .env loading

            assert settings.app_name == "Claude Bridge"
            assert settings.debug is False
            assert settings.host == "0.0.0.0"
            assert settings.port == 8080
            assert settings.log_level == "info"
            assert settings.log_json is False
            assert settings.claude_cli_path == "claude"
            assert settings.claude_cwd is None
            assert settings.claude_allowed_tools is None
            assert settings.claude_disallowed_tools is None
            assert settings.claude_process_timeout_seconds == 300
            assert settings.claude_stream_idle_timeout_seconds == 180
            assert settings.claude_process_kill_grace_seconds == 5

    def test_environment_variable_override(self):
        with patch.dict(
            "os.environ",
            {
                "APP_NAME": "Test App",
                "DEBUG": "true",
                "HOST": "127.0.0.1",
                "PORT": "9000",
                "LOG_LEVEL": "debug",
                "CLAUDE_CLI_PATH": "/usr/local/bin/claude",
            },
        ):
            settings = Settings()

            assert settings.app_name == "Test App"
            assert settings.debug is True
            assert settings.host == "127.0.0.1"
            assert settings.port == 9000
            assert settings.log_level == "debug"
            assert settings.claude_cli_path == "/usr/local/bin/claude"

    def test_claude_cwd_path_handling(self):
        with patch.dict("os.environ", {"CLAUDE_CWD": "/tmp/test"}):
            settings = Settings()
            assert settings.claude_cwd == Path("/tmp/test")

    def test_tool_permissions_list(self):
        with patch.dict(
            "os.environ",
            {
                "CLAUDE_ALLOWED_TOOLS": '["Read", "Write", "Bash"]',
                "CLAUDE_DISALLOWED_TOOLS": '["Bash"]',
            },
        ):
            settings = Settings()
            assert settings.claude_allowed_tools == ["Read", "Write", "Bash"]
            assert settings.claude_disallowed_tools == ["Bash"]

    def test_timeout_settings(self):
        with patch.dict(
            "os.environ",
            {
                "CLAUDE_PROCESS_TIMEOUT_SECONDS": "600",
                "CLAUDE_STREAM_IDLE_TIMEOUT_SECONDS": "300",
                "CLAUDE_PROCESS_KILL_GRACE_SECONDS": "10",
            },
        ):
            settings = Settings()
            assert settings.claude_process_timeout_seconds == 600
            assert settings.claude_stream_idle_timeout_seconds == 300
            assert settings.claude_process_kill_grace_seconds == 10


class TestLoggingConfiguration:
    def test_configure_logging_info_level(self):
        settings = Settings(log_level="info")

        # Clear existing handlers
        logging.getLogger().handlers = []

        settings.configure_logging()

        assert logging.getLogger().level == logging.INFO

    def test_configure_logging_debug_level(self):
        settings = Settings(log_level="debug")

        # Clear existing handlers
        logging.getLogger().handlers = []

        settings.configure_logging()

        assert logging.getLogger().level == logging.DEBUG

    def test_configure_logging_warning_level(self):
        settings = Settings(log_level="warning")

        # Clear existing handlers
        logging.getLogger().handlers = []

        settings.configure_logging()

        assert logging.getLogger().level == logging.WARNING

    def test_configure_logging_error_level(self):
        settings = Settings(log_level="error")

        # Clear existing handlers
        logging.getLogger().handlers = []

        settings.configure_logging()

        assert logging.getLogger().level == logging.ERROR

    def test_configure_logging_invalid_level_defaults_to_info(self):
        settings = Settings(log_level="invalid")

        # Clear existing handlers
        logging.getLogger().handlers = []

        settings.configure_logging()

        assert logging.getLogger().level == logging.INFO

    def test_configure_logging_skips_if_handlers_exist(self):
        settings = Settings(log_level="debug")

        # Add a handler to simulate existing configuration
        handler = logging.StreamHandler()
        logging.getLogger().addHandler(handler)
        initial_handler_count = len(logging.getLogger().handlers)

        settings.configure_logging()

        # Should not add new handlers
        assert len(logging.getLogger().handlers) == initial_handler_count

        # Clean up
        logging.getLogger().removeHandler(handler)
