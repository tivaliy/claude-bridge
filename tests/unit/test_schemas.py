"""Unit tests for Pydantic schemas."""

import pytest
from pydantic import ValidationError

from claude_bridge.anthropic.schemas import (
    ContentBlock,
    Message,
    MessagesRequest,
    MessagesResponse,
    Usage,
)


class TestMessage:
    def test_valid_user_message_string_content(self):
        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_valid_assistant_message(self):
        msg = Message(role="assistant", content="Hi there!")
        assert msg.role == "assistant"
        assert msg.content == "Hi there!"

    def test_valid_list_content(self):
        content = [{"type": "text", "text": "Hello"}]
        msg = Message(role="user", content=content)
        # Content is parsed into ContentBlock objects
        assert len(msg.content) == 1
        assert msg.content[0].type == "text"
        assert msg.content[0].text == "Hello"

    def test_invalid_role(self):
        with pytest.raises(ValidationError):
            Message(role="system", content="Hello")  # type: ignore


class TestMessagesRequest:
    def test_valid_minimal_request(self):
        req = MessagesRequest(
            model="claude-sonnet-4",
            messages=[Message(role="user", content="Hello")],
        )
        assert req.model == "claude-sonnet-4"
        assert len(req.messages) == 1
        assert req.max_tokens == 1024  # Default
        assert req.stream is False  # Default

    def test_valid_complete_request(self):
        req = MessagesRequest(
            model="claude-sonnet-4",
            messages=[Message(role="user", content="Hello")],
            max_tokens=2048,
            system="You are helpful.",
            stream=True,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
        )
        assert req.max_tokens == 2048
        assert req.system == "You are helpful."
        assert req.stream is True
        assert req.temperature == 0.7
        assert req.top_p == 0.9
        assert req.top_k == 50

    def test_max_tokens_minimum_constraint(self):
        with pytest.raises(ValidationError):
            MessagesRequest(
                model="claude-sonnet-4",
                messages=[Message(role="user", content="Hello")],
                max_tokens=0,
            )

    def test_temperature_range_constraint(self):
        with pytest.raises(ValidationError):
            MessagesRequest(
                model="claude-sonnet-4",
                messages=[Message(role="user", content="Hello")],
                temperature=1.5,
            )

        with pytest.raises(ValidationError):
            MessagesRequest(
                model="claude-sonnet-4",
                messages=[Message(role="user", content="Hello")],
                temperature=-0.1,
            )

    def test_top_p_range_constraint(self):
        with pytest.raises(ValidationError):
            MessagesRequest(
                model="claude-sonnet-4",
                messages=[Message(role="user", content="Hello")],
                top_p=1.1,
            )

    def test_top_k_minimum_constraint(self):
        with pytest.raises(ValidationError):
            MessagesRequest(
                model="claude-sonnet-4",
                messages=[Message(role="user", content="Hello")],
                top_k=-1,
            )

    def test_missing_required_fields(self):
        with pytest.raises(ValidationError):
            MessagesRequest(messages=[Message(role="user", content="Hello")])  # type: ignore

        with pytest.raises(ValidationError):
            MessagesRequest(model="claude-sonnet-4")  # type: ignore


class TestContentBlock:
    def test_valid_text_block(self):
        block = ContentBlock(type="text", text="Hello world")
        assert block.type == "text"
        assert block.text == "Hello world"

    def test_missing_text_field(self):
        with pytest.raises(ValidationError):
            ContentBlock(type="text")  # type: ignore


class TestUsage:
    def test_valid_usage(self):
        usage = Usage(input_tokens=100, output_tokens=200)
        assert usage.input_tokens == 100
        assert usage.output_tokens == 200

    def test_missing_fields(self):
        with pytest.raises(ValidationError):
            Usage(input_tokens=100)  # type: ignore

        with pytest.raises(ValidationError):
            Usage(output_tokens=200)  # type: ignore


class TestMessagesResponse:
    def test_valid_response(self):
        response = MessagesResponse(
            id="msg_abc123",
            type="message",
            role="assistant",
            content=[ContentBlock(type="text", text="Hello!")],
            model="claude-sonnet-4",
            stop_reason="end_turn",
            usage=Usage(input_tokens=10, output_tokens=5),
        )
        assert response.id == "msg_abc123"
        assert response.type == "message"
        assert response.role == "assistant"
        assert len(response.content) == 1
        assert response.model == "claude-sonnet-4"
        assert response.stop_reason == "end_turn"

    def test_default_values(self):
        response = MessagesResponse(
            id="msg_test",
            content=[ContentBlock(type="text", text="Hi")],
            model="claude-sonnet-4",
            usage=Usage(input_tokens=5, output_tokens=3),
        )
        assert response.type == "message"
        assert response.role == "assistant"
        assert response.stop_reason is None

    def test_stop_reason_validation(self):
        for stop_reason in ["end_turn", "max_tokens", "stop_sequence"]:
            response = MessagesResponse(
                id="msg_test",
                content=[ContentBlock(type="text", text="Hi")],
                model="claude-sonnet-4",
                stop_reason=stop_reason,  # type: ignore
                usage=Usage(input_tokens=5, output_tokens=3),
            )
            assert response.stop_reason == stop_reason
