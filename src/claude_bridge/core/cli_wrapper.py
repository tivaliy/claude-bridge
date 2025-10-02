"""Claude Code CLI wrapper for subprocess invocation."""

import asyncio
import contextlib
import json
import logging
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from ..config import settings


class ClaudeCLIWrapper:
    """Wrapper for invoking Claude Code CLI as a subprocess."""

    def __init__(
        self,
        cli_path: str = "claude",
        cwd: Path | None = None,
    ):
        """
        Initialize Claude CLI wrapper.

        Args:
            cli_path: Path to claude CLI binary (default: "claude")
            cwd: Working directory for CLI execution
        """
        self.cli_path = cli_path
        self.cwd = str(cwd) if cwd else None

    async def query(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        model: str | None = None,
        allowed_tools: list[str] | None = None,
        disallowed_tools: list[str] | None = None,
        stream: bool = False,
        file_paths: list[Path] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Execute a query using Claude CLI in non-interactive mode.

        Args:
            prompt: User prompt text
            system_prompt: Optional system prompt
            model: Model name/alias
            allowed_tools: Tools to allow
            disallowed_tools: Tools to disallow
            stream: Enable streaming mode
            file_paths: List of file paths for Claude to analyze

        Yields:
            Parsed JSON objects from CLI output (iterative streaming or single result).
        """
        logger = logging.getLogger("claude_bridge.cli_wrapper")
        start_time = time.perf_counter()

        # Build command
        cmd = [self.cli_path, "--print"]

        # Use bypassPermissions mode for non-interactive API usage
        # This is more targeted than --dangerously-skip-permissions
        # Combined with --allowed-tools and --add-dir, this provides minimal necessary access
        cmd.extend(["--permission-mode", "bypassPermissions"])

        # Add output format
        if stream:
            cmd.extend(
                [
                    "--verbose",
                    "--output-format",
                    "stream-json",
                    "--include-partial-messages",  # Enable true real-time streaming!
                ]
            )
        else:
            cmd.extend(["--output-format", "json"])

        # Add model if specified
        if model:
            cmd.extend(["--model", model])

        # Add system prompt
        if system_prompt:
            cmd.extend(["--append-system-prompt", system_prompt])

        # Add tool permissions from configuration
        # If file_paths are provided, validate configuration and use settings
        if file_paths:
            # Check if tools configured
            if not settings.claude_allowed_tools_str.strip():
                raise ValueError(
                    "File upload attempted but CLAUDE_ALLOWED_TOOLS_STR not configured. "
                    "Set CLAUDE_ALLOWED_TOOLS_STR=Read in your .env file to enable file upload."
                )

            # Check if directories configured
            if not settings.claude_allowed_directories_str.strip():
                raise ValueError(
                    "File upload attempted but CLAUDE_ALLOWED_DIRECTORIES_STR not configured. "
                    "Set CLAUDE_ALLOWED_DIRECTORIES_STR=/tmp (or your temp directory) in your .env file."
                )

            # Use configured tools
            allowed_tools_list = [
                t.strip() for t in settings.claude_allowed_tools_str.split(",") if t.strip()
            ]
            cmd.extend(["--allowed-tools", " ".join(allowed_tools_list)])
            logger.debug(f"Enabled tools from config: {allowed_tools_list}")

            # Use configured directories (NO auto-detection)
            allowed_dirs = [
                d.strip() for d in settings.claude_allowed_directories_str.split(",") if d.strip()
            ]
            for dir_path in allowed_dirs:
                cmd.extend(["--add-dir", dir_path])
            logger.debug(f"Allowed directories from config: {allowed_dirs}")

        # Legacy support for programmatic allowed_tools parameter
        elif allowed_tools:
            cmd.extend(["--allowed-tools", " ".join(allowed_tools)])

        # Disallowed tools from config or parameter
        disallowed_list = []
        if settings.claude_disallowed_tools_str.strip():
            disallowed_list.extend(
                [t.strip() for t in settings.claude_disallowed_tools_str.split(",") if t.strip()]
            )
        if disallowed_tools:
            disallowed_list.extend(disallowed_tools)

        if disallowed_list:
            cmd.extend(["--disallowed-tools", " ".join(disallowed_list)])

        # Build enhanced prompt with file references
        # Claude CLI can read files mentioned in the prompt using the Read tool
        enhanced_prompt = prompt
        if file_paths:
            file_instructions = "\n\n".join(
                [
                    f"Please use your Read tool to analyze this file: {file_path.absolute()}"
                    for file_path in file_paths
                ]
            )
            enhanced_prompt = f"{file_instructions}\n\n{prompt}"
            logger.debug(
                "Added file references to prompt",
                extra={"file_count": len(file_paths), "files": [str(f) for f in file_paths]},
            )

        # Don't add prompt as argument - we'll send via stdin for Read tool to work
        # cmd.append(enhanced_prompt)  # REMOVED

        # Execute subprocess
        logger.debug(
            "Launching CLI process",
            extra={
                "cmd": cmd,
                "stream": stream,
                "cwd": self.cwd,
                "model": model,
                "prompt_preview": enhanced_prompt[:100] if enhanced_prompt else "",
            },
        )

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,  # ADDED: Need stdin for prompt
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self.cwd,
        )

        timeout_seconds = settings.claude_process_timeout_seconds
        idle_timeout = settings.claude_stream_idle_timeout_seconds
        kill_grace = settings.claude_process_kill_grace_seconds

        async def ensure_timeout(proc: asyncio.subprocess.Process):
            try:
                await asyncio.wait_for(proc.wait(), timeout=timeout_seconds)
            except TimeoutError:
                logger.warning(
                    "CLI process timeout exceeded; terminating", extra={"timeout": timeout_seconds}
                )
                with contextlib.suppress(ProcessLookupError):
                    proc.terminate()
                try:
                    await asyncio.wait_for(proc.wait(), timeout=kill_grace)
                except TimeoutError:
                    with contextlib.suppress(ProcessLookupError):
                        proc.kill()

        # For non-streaming we will race process completion against timeout
        if not stream:
            # Fire-and-forget timeout enforcement (non-streaming)
            asyncio.create_task(ensure_timeout(process))

        # Read output
        if stream:
            # Send prompt to stdin for streaming
            if enhanced_prompt and process.stdin:
                process.stdin.write(enhanced_prompt.encode("utf-8"))
                await process.stdin.drain()
                process.stdin.close()

            last_line_time = time.monotonic()
            async for line in self._read_stream(process.stdout):
                if line.strip():
                    last_line_time = time.monotonic()
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        logger.debug("Skipping non-JSON stream line", extra={"line": line[:200]})
                        continue

                # Idle timeout check
                if (time.monotonic() - last_line_time) > idle_timeout:
                    logger.warning(
                        "Streaming idle timeout exceeded; terminating process",
                        extra={"idle_timeout": idle_timeout},
                    )
                    with contextlib.suppress(ProcessLookupError):
                        process.terminate()
                    break

            # Ensure process ended
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(process.wait(), timeout=kill_grace)
        else:
            # Read all output at once with timeout enforcement
            try:
                # Send prompt via stdin
                stdin_data = enhanced_prompt.encode("utf-8") if enhanced_prompt else None
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(input=stdin_data), timeout=timeout_seconds
                )
            except TimeoutError as e:
                logger.error(
                    "Non-streaming CLI timeout exceeded", extra={"timeout": timeout_seconds}
                )
                with contextlib.suppress(ProcessLookupError):
                    process.terminate()
                raise RuntimeError(
                    f"Claude CLI timed out after {timeout_seconds}s (terminated)"
                ) from e

            if process.returncode != 0:
                stderr_msg = stderr.decode() if stderr else ""
                stdout_msg = stdout.decode() if stdout else ""

                # Try to parse stdout as JSON first to extract proper error message
                error_msg = None
                if stdout_msg:
                    try:
                        result = json.loads(stdout_msg)
                        if result.get("is_error", False):
                            # Extract the error message from the result field
                            error_msg = result.get("result", "Unknown CLI error")
                    except json.JSONDecodeError:
                        # stdout is not JSON, use it as-is
                        error_msg = stdout_msg.strip()

                # Fall back to stderr if we couldn't extract from stdout
                if not error_msg and stderr_msg:
                    error_msg = stderr_msg.strip()

                # Final fallback
                if not error_msg:
                    error_msg = f"Command failed with exit code {process.returncode}"

                logger.error(
                    "CLI process failed",
                    extra={
                        "exit_code": process.returncode,
                        "stderr": stderr_msg[:500] if stderr_msg else None,
                        "stdout_preview": stdout_msg[:200] if stdout_msg else None,
                    },
                )
                raise RuntimeError(f"Claude CLI failed (exit {process.returncode}): {error_msg}")

            if stdout:
                try:
                    result = json.loads(stdout.decode())
                    # Check if CLI returned an error in the JSON
                    if result.get("is_error", False):
                        error_msg = result.get("result", "Unknown CLI error")
                        logger.warning("CLI returned is_error flag", extra={"error": error_msg})
                        raise RuntimeError(f"Claude CLI error: {error_msg}")
                    yield result
                except json.JSONDecodeError as e:
                    stdout_preview = stdout.decode()[:500]
                    logger.error(
                        "Failed to parse CLI output",
                        extra={"error": str(e), "stdout_preview": stdout_preview},
                    )
                    raise RuntimeError(
                        f"Failed to parse CLI output: {e}. Output preview: {stdout_preview}"
                    ) from e

        duration = time.perf_counter() - start_time
        logger.debug(
            "CLI invocation complete", extra={"duration_sec": round(duration, 4), "stream": stream}
        )

    async def _read_stream(self, stream) -> AsyncIterator[str]:
        """
        Read lines from async stream.

        Args:
            stream: Async stream to read from

        Yields:
            Individual lines from the stream
        """
        while True:
            line = await stream.readline()
            if not line:
                break
            yield line.decode()

    async def check_available(self) -> bool:
        """
        Check if Claude CLI is available.

        Returns:
            True if CLI is available, False otherwise
        """
        try:
            process = await asyncio.create_subprocess_exec(
                self.cli_path,
                "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await process.communicate()
            return process.returncode == 0
        except FileNotFoundError:
            return False

    async def get_version(self) -> str | None:
        """
        Get Claude CLI version.

        Returns:
            Version string or None if not available
        """
        try:
            process = await asyncio.create_subprocess_exec(
                self.cli_path,
                "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await process.communicate()
            if process.returncode == 0:
                return stdout.decode().strip()
            return None
        except FileNotFoundError:
            return None
