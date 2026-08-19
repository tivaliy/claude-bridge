# Claude Bridge

API gateway that intercepts Claude Code CLI and exposes it through Anthropic Messages API.

## Overview

Claude Bridge wraps the Claude Code CLI (not the SDK) and exposes it through an Anthropic-compatible REST API. It acts as an interceptor, translating HTTP API requests into CLI subprocess calls. Each request is handled as a single, stateless task to Claude Code.

Run it with an `ANTHROPIC_API_KEY` in the environment. The CLI can also run on a
Pro/Max subscription token, but Anthropic's terms reserve that credential for ordinary
interactive use of Claude Code rather than for fronting an API — see
[Authentication](#authentication) before choosing.

## Features

- **CLI Interceptor** - Wraps Claude Code CLI, not the Python SDK
- **Anthropic Messages API compatibility** - Drop-in replacement for Anthropic API
- **Anthropic SDK compatible** - Works seamlessly with the official Anthropic Python SDK
- **Streaming support** - Server-Sent Events (SSE) for real-time responses
- **Stateless architecture** - Each request is independent
- **No SDK dependency** - Direct subprocess calls to `claude` CLI
- **Locked down by default** - Tools that execute, mutate, delegate, persist, or reach
  the network are withheld, and no MCP servers are loaded (see [Security Implications](#2-understand-security-implications))

## Quick Start (No Installation Required)

Run Claude Bridge instantly with `uvx` (no installation needed):

```bash
uvx claude-bridge
```

This will download, install, and run the server in an isolated environment.

> **Note:** `uvx` requires the package to be published to PyPI. For local development, use `uv run claude-bridge` instead.

---

## Installation

### Prerequisites

- **Python 3.12+**
- **uv** (recommended) - Fast Python package installer and runner: `pip install uv`
- **Claude CLI** - Install from [claude.com](https://claude.com)
  - The `claude` command must be available in your PATH
- **An Anthropic API key** - `export ANTHROPIC_API_KEY=sk-ant-...`
  - See [Authentication](#authentication) for why this, and not `claude setup-token`,
    is the supported way to run the bridge

### Option 1: Run with uvx (Recommended - No Installation)

```bash
# Run directly without installing
uvx claude-bridge
```

### Option 2: Install Locally

Using `uv` (recommended):

```bash
uv pip install -e .
```

Or for development:

```bash
uv pip install -e ".[dev]"
```

This will install the `claude-bridge` CLI command in your virtual environment.

Alternatively, using `pip`:

```bash
pip install -e ".[dev]"
```

## Usage

### Start the server

**With uvx (no installation required):**

```bash
uvx claude-bridge
```

**After local installation:**

```bash
claude-bridge
```

**Or with `uv run`:**

```bash
uv run claude-bridge
```

**Or directly with Python:**

```bash
python main.py
```

The server will start on `http://localhost:8000` by default.

### Configuration

Create a `.env` file in the project root (or use environment variables):

```env
# Application settings
DEBUG=false
HOST=0.0.0.0
PORT=8000

# Claude CLI settings
CLAUDE_CLI_PATH=claude  # Path to claude binary
CLAUDE_CWD=/path/to/working/directory
```

---

## ⚠️ File Upload Configuration (Required)

File upload is **disabled by default** for security. To enable:

### 1. Configure via `.env` file OR command-line arguments

**Option A: Using `.env` file (persistent)**

```env
# Required for file upload
CLAUDE_ALLOWED_TOOLS_STR=Read
CLAUDE_ALLOWED_DIRECTORIES_STR=/tmp

# Permission mode (required for non-interactive API)
CLAUDE_PERMISSION_MODE=bypassPermissions
```

**Option B: Using CLI arguments (override .env, higher priority)**

```bash
claude-bridge --allowed-tools Read --allowed-directories /tmp
```

**Note:** CLI arguments take precedence over environment variables.

### 2. Understand Security Implications

**What you're allowing:**
- `CLAUDE_ALLOWED_TOOLS_STR=Read` - pre-approves the Read tool for file upload
- `CLAUDE_ALLOWED_DIRECTORIES_STR=/tmp` - the directories Claude may reach
- `CLAUDE_PERMISSION_MODE=bypassPermissions` - No interactive permission prompts (required for API mode). Informational only: the bridge always passes this mode regardless of the setting.

**What actually protects you — `CLAUDE_DISALLOWED_TOOLS_STR`, not the allow-list.**

Under `bypassPermissions` the allow-list pre-approves the tools you name but does
not withhold the rest: with `--allowed-tools Read`, Bash and Write are still
reachable. Only the deny-list withholds a tool. The bridge therefore ships a
secure default (`DEFAULT_DISALLOWED_TOOLS` in `config.py`) that withholds every
tool which executes, mutates, delegates to a subagent, outlives the request, or
reaches the network, leaving Read/Glob/Grep. It also passes `--strict-mcp-config`
so no MCP server is loaded — otherwise the CLI inherits the operator's own MCP
configuration and an HTTP caller reaches whatever that exposes.

**What is protected (with the defaults):**
- ❌ Claude **cannot** execute code, write files, or browse the web
- ❌ Claude **cannot** spawn subagents, schedule cron jobs, or send messages
- ❌ Claude **cannot** reach your MCP servers

**What is NOT protected — file reads are not confined to any directory.** Under
`bypassPermissions` the CLI applies no path restriction to `Read`/`Glob`/`Grep`, and
`--add-dir` only *adds* directories rather than limiting them. A caller can read any
file the server process can read — `.env`, `~/.claude/.credentials.json`, SSH and cloud
credentials — and have the contents returned in the response body. Run the bridge as a
user that cannot read anything sensitive, and do not expose the port.

**Also not protected:** the deny-list only names tools that exist today. A CLI upgrade
that adds a tool grants it to every caller of this API until it is listed. Re-check
after upgrading. Treat every prompt that reaches this API as untrusted input —
instructions hidden in an uploaded document are indistinguishable from the
caller's own.

### 3. Adjust for Your Environment

**macOS/Linux:**
```env
CLAUDE_ALLOWED_DIRECTORIES_STR=/tmp
```

**Windows:**
```env
CLAUDE_ALLOWED_DIRECTORIES_STR=C:\Temp
```

**Custom temp directory:**
```env
CLAUDE_ALLOWED_DIRECTORIES_STR=/var/myapp/temp
```

**Multiple directories (comma-separated):**
```env
CLAUDE_ALLOWED_DIRECTORIES_STR=/tmp,/var/app-temp
```

### 4. Start Server

After configuring `.env` or using CLI arguments:

```bash
claude-bridge
```

You should see:
```
INFO: Claude CLI permissions configured: tools=[Read], directories=[/tmp], mode=bypassPermissions
INFO: File upload is ENABLED
```

### Troubleshooting

**Error: "CLAUDE_ALLOWED_TOOLS_STR not configured"**
→ Add `CLAUDE_ALLOWED_TOOLS_STR=Read` to your `.env` file, OR
→ Use CLI argument: `--allowed-tools Read`

**Error: "contains non-existent directory"**
→ Verify the directory exists: `ls -la /tmp`

### Advanced: Using All Permission Arguments

```bash
# Enable multiple tools and directories via CLI
claude-bridge \
  --allowed-tools "Read,Bash" \
  --disallowed-tools "Write,Edit" \
  --allowed-directories "/tmp,/var/app-data"
```

**Permission errors in responses**
→ Ensure your temp directory matches where files are created

### Example: Upload File

```bash
curl -X POST http://localhost:8000/anthropic/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-sonnet-4-5-20250929",
    "max_tokens": 1024,
    "messages": [{
      "role": "user",
      "content": [{
        "type": "text",
        "text": "What is in this image?"
      }, {
        "type": "image",
        "source": {
          "type": "base64",
          "media_type": "image/png",
          "data": "iVBORw0KGgoAAAA...base64_image_data..."
        }
      }]
    }]
  }'
```

---

#### Authentication

The bridge does not authenticate anything itself — it inherits whatever credentials
the `claude` CLI runs under. The CLI accepts two, and **which one you use determines
whether running this bridge is permitted at all.**

##### Supported: API key

```bash
export ANTHROPIC_API_KEY=sk-ant-...
claude-bridge
```

Get a key from the [Claude Console](https://platform.claude.com/). Usage bills to
that key and falls under Anthropic's
[Commercial Terms](https://www.anthropic.com/legal/commercial-terms), which is the
agreement that covers programmatic access. `apiKeyHelper` works too if you resolve
keys dynamically. This is the mode to use for anything that is not a single person
poking at the bridge on their own laptop — any deployment, any shared instance,
anything backing an application, anything with more than one user.

##### Not supported for applications: subscription OAuth

The CLI can also authenticate against a Claude Pro/Max subscription via
`claude setup-token`, and the bridge will happily run that way. Anthropic's
[Claude Code legal terms](https://code.claude.com/docs/en/legal-and-compliance)
are explicit about what that credential is for:

> **OAuth authentication** is intended **exclusively** for purchasers of Claude Free,
> Pro, Max, Team, and Enterprise subscription plans and is designed to support
> **ordinary use of Claude Code** and other native Anthropic applications.
>
> **Developers** building products or services that interact with Claude's
> capabilities, including those using the Agent SDK, **should use API key
> authentication** through Claude Console or a supported cloud provider. Anthropic
> does not permit third-party developers to offer Claude.ai login or to **route
> requests through Free, Pro, or Max plan credentials on behalf of their users**.

Two consequences worth being clear about:

- Fronting a subscription token with an HTTP API is not "ordinary use of Claude
  Code," even with one user on one machine. If you do it, understand that is the
  posture you are in.
- Standing this up for anyone but yourself — a team instance, a hosted endpoint, a
  product backend — is the case the terms name outright. Don't.

Anthropic "reserves the right to take measures to enforce these restrictions and may
do so without prior notice," and enforcement lands on the subscription account. Set
`ANTHROPIC_API_KEY` and the question does not arise.

The bridge logs a warning at startup when it finds no API key in the environment.

##### Verify either way

```bash
claude --version
claude --print "Hello, Claude!"
```

### API Endpoints

#### POST /anthropic/v1/messages

Create a message using Claude Code.

**Supported Model Names:**

The bridge automatically maps Anthropic API model names to Claude CLI aliases:

- `claude-sonnet-4` → `sonnet` (latest Sonnet)
- `claude-opus-4` → `opus` (latest Opus)
- `claude-haiku-4` → `haiku` (latest Haiku)
- Or use full model names like `claude-sonnet-4-5-20250929` (recommended for stability)

**Non-streaming request:**

```bash
curl -X POST http://localhost:8000/anthropic/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-sonnet-4-5-20250929",
    "max_tokens": 1024,
    "messages": [
      {"role": "user", "content": "Hello, Claude!"}
    ]
  }'
```

**Streaming request:**

```bash
curl -X POST http://localhost:8000/anthropic/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-sonnet-4-5-20250929",
    "max_tokens": 1024,
    "stream": true,
    "messages": [
      {"role": "user", "content": "Count to 10"}
    ]
  }'
```

**With system prompt:**

```bash
curl -X POST http://localhost:8000/anthropic/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-sonnet-4-5-20250929",
    "max_tokens": 1024,
    "system": "You are a helpful assistant.",
    "messages": [
      {"role": "user", "content": "What is 2+2?"}
    ]
  }'
```

#### GET /health

Health check endpoint.

```bash
curl http://localhost:8000/health
```

### Using with Anthropic SDK

The bridge is fully compatible with the official Anthropic Python SDK. Simply configure the client with a custom `base_url`:

```python
import anthropic

client = anthropic.AsyncAnthropic(
    api_key="not-needed",  # Not checked by the bridge; the CLI holds the real credential
    base_url="http://localhost:8000/anthropic"
)

# Streaming example
async with client.messages.stream(
    model="claude-sonnet-4-5-20250929",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello!"}]
) as stream:
    async for text in stream.text_stream:
        print(text, end="", flush=True)

# Non-streaming example
message = await client.messages.create(
    model="claude-sonnet-4-5-20250929",
    max_tokens=1024,
    messages=[{"role": "user", "content": "What is 2+2?"}]
)
print(message.content[0].text)
```

**Note:** The bridge does not validate the `api_key` you pass here, and it is not an
access control — anything that can reach the port can spend your credentials. Bind it
to localhost, or put your own auth in front of it. The real credential is whichever one
the `claude` CLI is running under; see [Authentication](#authentication).

## Development

### Setup Pre-commit Hooks

This project uses pre-commit hooks for code quality and consistency:

```bash
# Install hooks (one-time setup)
pre-commit install

# Run manually on all files (optional)
pre-commit run --all-files
```

Hooks will automatically run on `git commit` and include:
- ✅ Code formatting (ruff)
- ✅ Linting (ruff)
- ✅ Type checking (mypy)
- ✅ Security scanning (bandit)
- ✅ File hygiene (trailing whitespace, EOF, etc.)
- ✅ Secret detection (detect-private-key)

### Run tests

The project includes comprehensive unit and integration tests:

```bash
# Run all tests with coverage
pytest

# Run specific test categories
pytest tests/unit/          # Unit tests only
pytest tests/integration/   # Integration tests only

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/unit/test_adapter.py

# Run without coverage report
pytest --no-cov

# Generate HTML coverage report
pytest --cov-report=html
# Then open htmlcov/index.html in your browser
```

**Test Structure:**
- `tests/unit/` - Fast, isolated unit tests for individual components
- `tests/integration/` - Integration tests with mocked external dependencies
- `tests/fixtures/` - Sample data for tests
- `tests/conftest.py` - Shared pytest fixtures

### Manual code quality checks

```bash
# Linting and formatting
ruff check src/ --fix
ruff format src/

# Type checking
mypy src/

# Security scanning
bandit -c pyproject.toml -r src/
```

## License

MIT
