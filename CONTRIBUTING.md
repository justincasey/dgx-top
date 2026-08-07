# Contributing to dgx-top

Thanks for improving dgx-top.

## Development setup

```bash
git clone https://github.com/justincasey/dgx-top.git
cd dgx-top
uv sync --extra dev
uv run pytest
```

## Before submitting a change

```bash
uv run ruff format .
uv run ruff check .
uv run pytest
uv build
```

Add tests for parsing, configuration, or failure behavior whenever those boundaries change. Keep remote collection agentless and preserve graceful handling when either SSH telemetry or vLLM metrics are unavailable.

## Privacy

Never commit real credentials, private keys, internal DNS names, private IP addresses, MAC addresses, device serial numbers, model paths containing usernames, or telemetry captured from a private system. Use reserved documentation domains such as `example.com` and synthetic test data.

Before attaching logs or screenshots to an issue, remove network identifiers, model names that may be sensitive, usernames, and filesystem paths.

## Pull requests

- Explain the user-visible behavior and why it is needed.
- Include the commands used for validation.
- Keep unrelated formatting or cleanup out of the change.
- Update README or architecture documentation when setup or boundaries change.

By contributing, you agree that your contribution is licensed under the MIT License.
