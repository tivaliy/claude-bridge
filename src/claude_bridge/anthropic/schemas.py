"""Pydantic models for Anthropic Messages API."""

from typing import Any, Literal

from pydantic import BaseModel, Field


# Content block input models (for requests)
class Base64ImageSource(BaseModel):
    """Base64 encoded image source."""

    type: Literal["base64"] = "base64"
    media_type: Literal["image/jpeg", "image/png", "image/gif", "image/webp"]
    data: str


class Base64PDFSource(BaseModel):
    """Base64 encoded PDF source."""

    type: Literal["base64"] = "base64"
    media_type: Literal["application/pdf"]
    data: str


class ImageContentBlock(BaseModel):
    """Image content block in message."""

    type: Literal["image"] = "image"
    source: Base64ImageSource


class DocumentContentBlock(BaseModel):
    """Document (PDF) content block in message."""

    type: Literal["document"] = "document"
    source: Base64PDFSource


class TextContentBlock(BaseModel):
    """Text content block in message."""

    type: Literal["text"] = "text"
    text: str


# Union type for content blocks
ContentBlockInput = ImageContentBlock | DocumentContentBlock | TextContentBlock


# Request models
class Message(BaseModel):
    """A message in the conversation."""

    role: Literal["user", "assistant"]
    content: str | list[ContentBlockInput]


class MessagesRequest(BaseModel):
    """Request body for /v1/messages endpoint."""

    model: str
    messages: list[Message]
    max_tokens: int = Field(default=1024, ge=1)
    metadata: dict[str, Any] | None = None
    stop_sequences: list[str] | None = None
    stream: bool = False
    system: str | None = None
    temperature: float | None = Field(default=None, ge=0.0, le=1.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    top_k: int | None = Field(default=None, ge=0)
    tools: list[dict[str, Any]] | None = None


# Response models
class ContentBlock(BaseModel):
    """A content block in the response."""

    type: Literal["text"]
    text: str


class Usage(BaseModel):
    """Token usage information."""

    input_tokens: int
    output_tokens: int


class MessagesResponse(BaseModel):
    """Response body for /v1/messages endpoint (non-streaming)."""

    id: str
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    content: list[ContentBlock]
    model: str
    stop_reason: Literal["end_turn", "max_tokens", "stop_sequence"] | None = None
    stop_sequence: str | None = None
    usage: Usage


# Streaming models
class StreamEvent(BaseModel):
    """Base class for streaming events."""

    type: str


class MessageStartEvent(StreamEvent):
    """Event when message starts."""

    type: Literal["message_start"] = "message_start"
    message: dict[str, Any]


class ContentBlockStartEvent(StreamEvent):
    """Event when content block starts."""

    type: Literal["content_block_start"] = "content_block_start"
    index: int
    content_block: ContentBlock


class ContentBlockDeltaEvent(StreamEvent):
    """Event for content block delta."""

    type: Literal["content_block_delta"] = "content_block_delta"
    index: int
    delta: dict[str, Any]


class ContentBlockStopEvent(StreamEvent):
    """Event when content block stops."""

    type: Literal["content_block_stop"] = "content_block_stop"
    index: int


class MessageDeltaEvent(StreamEvent):
    """Event for message delta."""

    type: Literal["message_delta"] = "message_delta"
    delta: dict[str, Any]
    usage: Usage | None = None


class MessageStopEvent(StreamEvent):
    """Event when message stops."""

    type: Literal["message_stop"] = "message_stop"


class PingEvent(StreamEvent):
    """Ping event."""

    type: Literal["ping"] = "ping"


class ErrorEvent(StreamEvent):
    """Error event."""

    type: Literal["error"] = "error"
    error: dict[str, Any]
