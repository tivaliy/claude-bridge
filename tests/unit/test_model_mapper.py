"""Unit tests for ModelMapper."""

from claude_bridge.core.model_mapper import ModelMapper


class TestModelMapper:
    def test_map_known_alias_sonnet(self):
        result = ModelMapper.map_model("claude-sonnet-4")
        assert result == "sonnet"

    def test_map_known_alias_opus(self):
        result = ModelMapper.map_model("claude-opus-4")
        assert result == "opus"

    def test_map_known_alias_haiku(self):
        result = ModelMapper.map_model("claude-haiku-4")
        assert result == "haiku"

    def test_map_claude_3_variants(self):
        assert ModelMapper.map_model("claude-3-5-sonnet") == "sonnet"
        assert ModelMapper.map_model("claude-3-opus") == "opus"
        assert ModelMapper.map_model("claude-3-haiku") == "haiku"

    def test_map_full_model_name_passthrough(self):
        full_name = "claude-sonnet-4-5-20250929"
        result = ModelMapper.map_model(full_name)
        assert result == full_name

    def test_map_full_model_name_sonnet(self):
        result = ModelMapper.map_model("claude-3-5-sonnet-20241022")
        assert result == "sonnet"

    def test_map_full_model_name_haiku(self):
        result = ModelMapper.map_model("claude-3-5-haiku-20241022")
        assert result == "haiku"

    def test_pattern_extraction_sonnet(self):
        result = ModelMapper.map_model("claude-sonnet-99-experimental")
        assert result == "sonnet"

    def test_pattern_extraction_opus(self):
        result = ModelMapper.map_model("claude-opus-future-version")
        assert result == "opus"

    def test_pattern_extraction_haiku(self):
        result = ModelMapper.map_model("claude-haiku-beta")
        assert result == "haiku"

    def test_unknown_model_passthrough(self):
        unknown_model = "gpt-4-turbo"
        result = ModelMapper.map_model(unknown_model)
        assert result == unknown_model

    def test_non_claude_model_passthrough(self):
        result = ModelMapper.map_model("some-other-model")
        assert result == "some-other-model"

    def test_empty_string(self):
        result = ModelMapper.map_model("")
        assert result == ""

    def test_case_sensitivity(self):
        # Uppercase version should not match exact map entry
        result = ModelMapper.map_model("CLAUDE-SONNET-4")
        # Pattern extraction is also case-sensitive, so returns as-is
        assert result == "CLAUDE-SONNET-4"

    def test_partial_match_priority(self):
        # Exact match in MODEL_MAP
        assert ModelMapper.map_model("claude-sonnet-4") == "sonnet"
        # Pattern extraction fallback
        assert ModelMapper.map_model("claude-4-sonnet") == "sonnet"
