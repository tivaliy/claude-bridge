"""Unit tests for AnthropicMessageAdapter."""

import pytest

from claude_bridge.anthropic.adapter import AnthropicMessageAdapter
from claude_bridge.anthropic.schemas import Message


class TestMessagesToPrompt:
    @pytest.mark.asyncio
    async def test_single_user_message_string_content(self):
        messages = [Message(role="user", content="Hello, Claude!")]
        prompt, system, _ = await AnthropicMessageAdapter.messages_to_prompt(messages)

        assert prompt == "Hello, Claude!"
        assert system is None

    @pytest.mark.asyncio
    async def test_single_user_message_with_system(self):
        messages = [Message(role="user", content="What is 2+2?")]
        system_prompt = "You are a helpful assistant."
        prompt, system, _ = await AnthropicMessageAdapter.messages_to_prompt(
            messages, system_prompt
        )

        assert prompt == "What is 2+2?"
        assert system == "You are a helpful assistant."

    @pytest.mark.asyncio
    async def test_multiple_messages_builds_context(self):
        messages = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi! How can I help?"),
            Message(role="user", content="Tell me a joke"),
        ]
        prompt, system, _ = await AnthropicMessageAdapter.messages_to_prompt(messages)

        expected = (
            "<conversation>\n"
            '<turn role="user">Hello</turn>\n'
            '<turn role="assistant">Hi! How can I help?</turn>\n'
            "</conversation>\n\n"
            "Tell me a joke"
        )
        assert prompt == expected
        assert system is None

    @pytest.mark.asyncio
    async def test_list_content_blocks(self):
        messages = [
            Message(
                role="user",
                content=[
                    {"type": "text", "text": "First part"},
                    {"type": "text", "text": "Second part"},
                ],
            )
        ]
        prompt, system, _ = await AnthropicMessageAdapter.messages_to_prompt(messages)

        assert prompt == "First part\nSecond part"

    @pytest.mark.asyncio
    async def test_list_content_with_non_text_blocks(self):
        # Use valid base64 string (1x1 transparent PNG)
        valid_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        messages = [
            Message(
                role="user",
                content=[
                    {"type": "text", "text": "Analyze this:"},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": valid_base64,
                        },
                    },
                    {"type": "text", "text": "What do you see?"},
                ],
            )
        ]
        prompt, system, file_paths = await AnthropicMessageAdapter.messages_to_prompt(messages)

        # Image blocks create temp files, text blocks are extracted
        assert "Analyze this:" in prompt
        assert "What do you see?" in prompt
        assert len(file_paths) == 1  # One image file

    @pytest.mark.asyncio
    async def test_empty_messages_list(self):
        prompt, system, _ = await AnthropicMessageAdapter.messages_to_prompt([])

        assert prompt == ""
        assert system is None

    @pytest.mark.asyncio
    async def test_no_user_messages(self):
        messages = [Message(role="assistant", content="Hello!")]
        prompt, system, _ = await AnthropicMessageAdapter.messages_to_prompt(messages)

        assert prompt == ""
        assert system is None

    @pytest.mark.asyncio
    async def test_empty_content_string(self):
        messages = [Message(role="user", content="")]
        prompt, system, _ = await AnthropicMessageAdapter.messages_to_prompt(messages)

        assert prompt == ""

    @pytest.mark.asyncio
    async def test_empty_content_list(self):
        messages = [Message(role="user", content=[])]
        prompt, system, _ = await AnthropicMessageAdapter.messages_to_prompt(messages)

        assert prompt == ""

    @pytest.mark.asyncio
    async def test_content_blocks_with_empty_text(self):
        messages = [
            Message(
                role="user",
                content=[
                    {"type": "text", "text": ""},  # Empty text
                    {"type": "text", "text": "Valid text"},
                ],
            )
        ]
        prompt, system, _ = await AnthropicMessageAdapter.messages_to_prompt(messages)

        # Should include empty string for empty text, then valid text
        assert prompt == "\nValid text"

    @pytest.mark.asyncio
    async def test_multi_turn_with_content_blocks(self):
        messages = [
            Message(role="user", content=[{"type": "text", "text": "My name is Bob"}]),
            Message(role="assistant", content="Hello Bob!"),
            Message(role="user", content=[{"type": "text", "text": "What's my name?"}]),
        ]
        prompt, system, file_paths = await AnthropicMessageAdapter.messages_to_prompt(messages)

        expected = (
            "<conversation>\n"
            '<turn role="user">My name is Bob</turn>\n'
            '<turn role="assistant">Hello Bob!</turn>\n'
            "</conversation>\n\n"
            "What's my name?"
        )
        assert prompt == expected
        assert file_paths == []

    @pytest.mark.asyncio
    async def test_multi_turn_with_system_prompt(self):
        messages = [
            Message(role="user", content="First question"),
            Message(role="assistant", content="First answer"),
            Message(role="user", content="Second question"),
        ]
        system_prompt = "You are a helpful assistant."
        prompt, system, _ = await AnthropicMessageAdapter.messages_to_prompt(
            messages, system_prompt
        )

        assert "<conversation>" in prompt
        assert '<turn role="user">First question</turn>' in prompt
        assert '<turn role="assistant">First answer</turn>' in prompt
        assert "Second question" in prompt
        assert system == "You are a helpful assistant."

    @pytest.mark.asyncio
    async def test_long_multi_turn_conversation(self):
        messages = [
            Message(role="user", content="Turn 1"),
            Message(role="assistant", content="Response 1"),
            Message(role="user", content="Turn 2"),
            Message(role="assistant", content="Response 2"),
            Message(role="user", content="Turn 3"),
            Message(role="assistant", content="Response 3"),
            Message(role="user", content="Final turn"),
        ]
        prompt, system, _ = await AnthropicMessageAdapter.messages_to_prompt(messages)

        assert "<conversation>" in prompt
        assert '<turn role="user">Turn 1</turn>' in prompt
        assert '<turn role="assistant">Response 1</turn>' in prompt
        assert '<turn role="user">Turn 2</turn>' in prompt
        assert '<turn role="assistant">Response 2</turn>' in prompt
        assert '<turn role="user">Turn 3</turn>' in prompt
        assert '<turn role="assistant">Response 3</turn>' in prompt
        assert prompt.endswith("Final turn")

    @pytest.mark.asyncio
    async def test_multi_turn_with_assistant_last(self):
        messages = [
            Message(role="user", content="First question"),
            Message(role="assistant", content="First response"),
            Message(role="user", content="Second question"),
            Message(role="assistant", content="Second response"),
        ]
        prompt, system, _ = await AnthropicMessageAdapter.messages_to_prompt(messages)

        # Last USER message is at index 2 ("Second question")
        # History includes messages before index 2: "First question" and "First response"
        expected = (
            "<conversation>\n"
            '<turn role="user">First question</turn>\n'
            '<turn role="assistant">First response</turn>\n'
            "</conversation>\n\n"
            "Second question"
        )
        assert prompt == expected

    @pytest.mark.asyncio
    async def test_invalid_base64_raises_error(self):
        messages = [
            Message(
                role="user",
                content=[
                    {"type": "text", "text": "Check this image"},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": "invalid-base64!!!",
                        },
                    },
                ],
            )
        ]

        with pytest.raises(ValueError, match="Failed to process image"):
            await AnthropicMessageAdapter.messages_to_prompt(messages)

    @pytest.mark.asyncio
    async def test_oversized_file_raises_error(self):
        # Create base64 that decodes to > 5MB (for images)
        large_data = "A" * (6 * 1024 * 1024)  # 6MB of 'A's
        import base64

        large_base64 = base64.b64encode(large_data.encode()).decode()

        messages = [
            Message(
                role="user",
                content=[
                    {"type": "text", "text": "Check this image"},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": large_base64,
                        },
                    },
                ],
            )
        ]

        with pytest.raises(ValueError, match="Failed to process image"):
            await AnthropicMessageAdapter.messages_to_prompt(messages)


class TestCLIResponseToAnthropic:
    async def test_success_response_conversion(self, make_async_iter):
        cli_response = make_async_iter(
            [
                {
                    "type": "result",
                    "subtype": "success",
                    "result": "Hello! How can I help?",
                    "usage": {"input_tokens": 10, "output_tokens": 15},
                    "is_error": False,
                }
            ]
        )

        response = await AnthropicMessageAdapter.cli_response_to_anthropic(
            cli_response, "claude-sonnet-4"
        )

        assert response.type == "message"
        assert response.role == "assistant"
        assert len(response.content) == 1
        assert response.content[0].type == "text"
        assert response.content[0].text == "Hello! How can I help?"
        assert response.model == "claude-sonnet-4"
        assert response.usage.input_tokens == 10
        assert response.usage.output_tokens == 15
        assert response.stop_reason == "end_turn"
        assert response.id.startswith("msg_")

    async def test_usage_extraction(self, make_async_iter):
        cli_response = make_async_iter(
            [
                {
                    "result": "Response text",
                    "usage": {"input_tokens": 100, "output_tokens": 200},
                }
            ]
        )

        response = await AnthropicMessageAdapter.cli_response_to_anthropic(
            cli_response, "claude-sonnet-4"
        )

        assert response.usage.input_tokens == 100
        assert response.usage.output_tokens == 200

    async def test_missing_usage_defaults_to_zero(self, make_async_iter):
        cli_response = make_async_iter([{"result": "Response text"}])

        response = await AnthropicMessageAdapter.cli_response_to_anthropic(
            cli_response, "claude-sonnet-4"
        )

        assert response.usage.input_tokens == 0
        assert response.usage.output_tokens == 0

    async def test_error_response_raises(self, make_async_iter):
        cli_response = make_async_iter([{"is_error": True, "result": "Model not found"}])

        with pytest.raises(RuntimeError, match="Model not found"):
            await AnthropicMessageAdapter.cli_response_to_anthropic(cli_response, "claude-sonnet-4")

    async def test_empty_response(self, make_async_iter):
        cli_response = make_async_iter([])

        response = await AnthropicMessageAdapter.cli_response_to_anthropic(
            cli_response, "claude-sonnet-4"
        )

        assert response.content[0].text == ""
        assert response.usage.input_tokens == 0
        assert response.usage.output_tokens == 0

    async def test_multiple_chunks_uses_last(self, make_async_iter):
        cli_response = make_async_iter(
            [
                {"result": "First chunk", "usage": {"input_tokens": 5, "output_tokens": 10}},
                {"result": "Final chunk", "usage": {"input_tokens": 10, "output_tokens": 20}},
            ]
        )

        response = await AnthropicMessageAdapter.cli_response_to_anthropic(
            cli_response, "claude-sonnet-4"
        )

        assert response.content[0].text == "Final chunk"
        assert response.usage.input_tokens == 10
        assert response.usage.output_tokens == 20

    async def test_message_id_generation(self, make_async_iter):
        cli_response1 = make_async_iter([{"result": "Text 1"}])
        cli_response2 = make_async_iter([{"result": "Text 2"}])

        response1 = await AnthropicMessageAdapter.cli_response_to_anthropic(
            cli_response1, "claude-sonnet-4"
        )
        response2 = await AnthropicMessageAdapter.cli_response_to_anthropic(
            cli_response2, "claude-sonnet-4"
        )

        assert response1.id != response2.id
        assert len(response1.id) == 28  # "msg_" + 24 hex chars
        assert len(response2.id) == 28
