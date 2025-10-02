"""Message adapter for converting between Anthropic format and Claude CLI format."""

import base64
import logging
import os
import tempfile
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from .schemas import ContentBlock, Message, MessagesResponse, Usage

logger = logging.getLogger("claude_bridge.adapter")


class ContentProcessor:
    """Handles content block processing including file extraction."""

    # Media type to file extension mapping
    MEDIA_TYPE_EXTENSIONS = {
        # Images
        "image/jpeg": ".jpg",
        "image/png": ".png",
        "image/gif": ".gif",
        "image/webp": ".webp",
        # Documents
        "application/pdf": ".pdf",
    }

    # Maximum file sizes (in bytes)
    MAX_FILE_SIZES = {
        "image/jpeg": 5 * 1024 * 1024,  # 5 MB
        "image/png": 5 * 1024 * 1024,  # 5 MB
        "image/gif": 5 * 1024 * 1024,  # 5 MB
        "image/webp": 5 * 1024 * 1024,  # 5 MB
        "application/pdf": 32 * 1024 * 1024,  # 32 MB
    }

    @classmethod
    def _get_file_extension(cls, media_type: str) -> str:
        """Map media type to file extension."""
        return cls.MEDIA_TYPE_EXTENSIONS.get(media_type, ".bin")

    @classmethod
    def _validate_file_size(cls, data_bytes: bytes, media_type: str) -> None:
        """Validate file size against maximum allowed."""
        max_size = cls.MAX_FILE_SIZES.get(media_type, 32 * 1024 * 1024)
        if len(data_bytes) > max_size:
            max_mb = max_size / (1024 * 1024)
            actual_mb = len(data_bytes) / (1024 * 1024)
            raise ValueError(
                f"File size ({actual_mb:.2f}MB) exceeds maximum allowed "
                f"for {media_type} ({max_mb:.2f}MB)"
            )

    @classmethod
    async def process_content_blocks(
        cls,
        content: str | list[dict[str, Any]] | list[Any],
    ) -> tuple[str, list[Path]]:
        """
        Process message content blocks, extracting text and creating temp files for media.

        Args:
            content: Message content (string or list of content blocks)

        Returns:
            Tuple of (prompt_text, temp_file_paths)
        """
        # Handle simple string content
        if isinstance(content, str):
            return content, []

        # Handle content blocks
        if not isinstance(content, list):
            return str(content), []

        text_parts = []
        temp_files = []

        for block in content:
            # Convert Pydantic models to dict if needed
            if hasattr(block, "model_dump"):
                block = block.model_dump()
            elif not isinstance(block, dict):
                continue

            block_type = block.get("type")

            if block_type == "text":
                text_parts.append(block.get("text", ""))

            elif block_type in ["image", "document"]:
                source = block.get("source", {})

                if source.get("type") == "base64":
                    media_type = source.get("media_type", "")
                    data = source.get("data", "")

                    try:
                        # Decode base64
                        decoded_bytes = base64.b64decode(data)

                        # Validate file size
                        cls._validate_file_size(decoded_bytes, media_type)

                        # Create temp file with proper permissions
                        ext = cls._get_file_extension(media_type)
                        with tempfile.NamedTemporaryFile(
                            suffix=ext, prefix="bridge_", delete=False
                        ) as tf:
                            tf.write(decoded_bytes)
                            temp_file = Path(tf.name)

                        # Set explicit read permissions for Claude CLI
                        os.chmod(temp_file, 0o644)  # rw-r--r--

                        temp_files.append(temp_file)

                        # Add reference to prompt
                        file_type = "Image" if block_type == "image" else "Document"
                        text_parts.append(f"[{file_type}: {temp_file}]")

                        logger.debug(
                            f"Created temp file for {block_type}",
                            extra={
                                "file": str(temp_file),
                                "media_type": media_type,
                                "size_bytes": len(decoded_bytes),
                                "permissions": "0o644",
                            },
                        )

                    except Exception as e:
                        logger.error(
                            f"Failed to process {block_type} content block",
                            exc_info=True,
                            extra={"media_type": media_type, "error": str(e)},
                        )
                        raise ValueError(f"Failed to process {block_type}: {e}") from e

        prompt = "\n".join(text_parts) if text_parts else ""
        return prompt, temp_files


class AnthropicMessageAdapter:
    """Handles conversion between Anthropic API format and Claude CLI JSON format."""

    @classmethod
    def _extract_text_content(cls, content: str | list[dict[str, Any]] | list[Any]) -> str:
        """
        Extract text content from message content (string or content blocks).

        Args:
            content: Message content (string or list of content blocks)

        Returns:
            Extracted text content
        """
        if isinstance(content, str):
            return content
        elif isinstance(content, list):
            text_parts = []
            for block in content:
                # Convert Pydantic models to dict if needed
                if hasattr(block, "model_dump"):
                    block = block.model_dump()

                if isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
            return "\n".join(text_parts)
        else:
            return str(content)

    @classmethod
    async def messages_to_prompt(
        cls,
        messages: list[Message],
        system: str | None = None,
    ) -> tuple[str, str | None, list[Path]]:
        """
        Convert Anthropic messages to Claude CLI prompt format with conversation context.

        For multi-turn conversations, formats the conversation history into the prompt
        to provide full context to the CLI in a single inference call.

        Args:
            messages: List of Anthropic Message objects
            system: Optional system prompt from request

        Returns:
            Tuple of (prompt, system_prompt, temp_files)
        """
        if not messages:
            logger.debug("No messages provided to adapter; returning empty prompt")
            return "", system, []

        # Find the index of the last user message
        last_user_index = None
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].role == "user":
                last_user_index = i
                break

        if last_user_index is None:
            logger.debug("No user role message found; returning empty prompt")
            return "", system, []

        # Process the final user message for files and text
        final_message_content = messages[last_user_index].content
        prompt_text, temp_files = await ContentProcessor.process_content_blocks(
            final_message_content
        )

        # Handle single message (no conversation history)
        if len(messages) == 1:
            logger.debug(
                "Single message request",
                extra={
                    "prompt_preview": prompt_text[:120],
                    "has_system": bool(system),
                    "temp_files": len(temp_files),
                },
            )
            return prompt_text, system, temp_files

        # Multi-turn conversation: build context from all messages before the last user message
        logger.info(
            "Multi-turn conversation detected - building conversation context",
            extra={"total_messages": len(messages)},
        )

        conversation_parts = []
        for i in range(last_user_index):  # All messages before last user message
            msg = messages[i]
            content = cls._extract_text_content(msg.content)
            conversation_parts.append(f'<turn role="{msg.role}">{content}</turn>')

        # Format: XML-structured conversation history + current question
        # Using XML tags as recommended by Anthropic for clear structure
        full_prompt = (
            "<conversation>\n"
            + "\n".join(conversation_parts)
            + "\n</conversation>\n\n"
            + prompt_text
        )

        logger.debug(
            "Built multi-turn prompt",
            extra={
                "history_turns": len(conversation_parts),
                "prompt_preview": full_prompt[:200],
                "has_system": bool(system),
                "temp_files": len(temp_files),
            },
        )

        return full_prompt, system, temp_files

    @classmethod
    async def cli_response_to_anthropic(
        cls,
        cli_response: AsyncIterator[dict[str, Any]],
        model: str,
    ) -> MessagesResponse:
        """
        Convert Claude CLI JSON response to Anthropic Messages API format (non-streaming).

        CLI JSON format (--output-format json):
        {
            "type": "result",
            "subtype": "success",
            "result": "response text",
            "usage": {"input_tokens": 10, "output_tokens": 20, ...},
            "is_error": false
        }

        Args:
            cli_response: AsyncIterator yielding CLI JSON responses
            model: Model name from request

        Returns:
            MessagesResponse object
        """
        content_text = ""
        input_tokens = 0
        output_tokens = 0

        # Collect all response content from CLI
        async for response in cli_response:
            if isinstance(response, dict):
                # Check for error
                if response.get("is_error", False):
                    error_msg = response.get("result", "Unknown CLI error")
                    logger.warning("CLI error flagged in adapter", extra={"error": error_msg})
                    raise RuntimeError(f"Claude CLI error: {error_msg}")

                # Extract result text from CLI response
                if "result" in response:
                    content_text = response["result"]

                # Extract usage if available
                if "usage" in response:
                    usage = response["usage"]
                    input_tokens = usage.get("input_tokens", 0)
                    output_tokens = usage.get("output_tokens", 0)

        logger.debug(
            "Assembled non-streaming response",
            extra={
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "content_len": len(content_text),
            },
        )

        # Generate response ID
        response_id = f"msg_{uuid.uuid4().hex[:24]}"

        # Create response
        return MessagesResponse(
            id=response_id,
            type="message",
            role="assistant",
            content=[ContentBlock(type="text", text=content_text)],
            model=model,
            stop_reason="end_turn",
            usage=Usage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            ),
        )
