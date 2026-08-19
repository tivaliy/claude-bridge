"""Unit tests for application startup checks."""

import logging
from unittest.mock import patch

from claude_bridge.app import _warn_if_no_api_key


class TestApiKeyWarning:
    def test_warns_when_no_api_key_is_set(self, caplog):
        logger = logging.getLogger("claude_bridge.app.test")

        with patch.dict("os.environ", {}, clear=True), caplog.at_level(logging.WARNING):
            _warn_if_no_api_key(logger)

        assert "ANTHROPIC_API_KEY" in caplog.text
        assert "legal-and-compliance" in caplog.text

    def test_silent_when_api_key_is_set(self, caplog):
        logger = logging.getLogger("claude_bridge.app.test")

        with (
            patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-ant-test"}, clear=True),
            caplog.at_level(logging.WARNING),
        ):
            _warn_if_no_api_key(logger)

        assert caplog.text == ""
