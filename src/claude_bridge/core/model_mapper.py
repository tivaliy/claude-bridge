"""Model name mapping between Anthropic API and Claude CLI."""


class ModelMapper:
    """Maps Anthropic API model names to Claude CLI model names."""

    # Mapping of common Anthropic API model names to CLI aliases/names
    MODEL_MAP = {
        # Aliases
        "sonnet": "sonnet",
        "opus": "opus",
        "haiku": "haiku",
        # Common full names to aliases
        "claude-sonnet-4": "sonnet",
        "claude-opus-4": "opus",
        "claude-haiku-4": "haiku",
        # Common variations
        "claude-3-5-sonnet": "sonnet",
        "claude-3-opus": "opus",
        "claude-3-haiku": "haiku",
        # Full model names (pass through unchanged)
        "claude-sonnet-4-5-20250929": "claude-sonnet-4-5-20250929",
    }

    @classmethod
    def map_model(cls, anthropic_model: str) -> str:
        """
        Map Anthropic API model name to Claude CLI model name.

        Args:
            anthropic_model: Model name from Anthropic API request

        Returns:
            Claude CLI-compatible model name (alias or full name)
        """
        # If it's in our map, use the mapping
        if anthropic_model in cls.MODEL_MAP:
            return cls.MODEL_MAP[anthropic_model]

        # If it starts with claude- and has specific patterns, try to extract alias
        if anthropic_model.startswith("claude-"):
            # Extract the model type (sonnet, opus, haiku)
            for model_type in ["sonnet", "opus", "haiku"]:
                if model_type in anthropic_model.lower():
                    return model_type

        # Default: pass through as-is (might be a valid full name)
        return anthropic_model
