# dgx-top

<p align="center">
  <img src="dgx-top-hero.jpeg" alt="dgx-top terminal dashboard" width="75%">
</p>

[![CI](https://github.com/justincasey/dgx-top/actions/workflows/ci.yml/badge.svg)](https://github.com/justincasey/dgx-top/actions/workflows/ci.yml)

`dgx-top` is an agentless terminal dashboard for monitoring one- or two-node NVIDIA DGX Spark clusters running vLLM. It combines hardware telemetry collected over SSH with vLLM's HTTP metrics—nothing is installed on the Spark nodes.

## What it shows

- Prompt and generated token throughput as separate time series
- Prompt-to-generation ratio
- Running and waiting requests
- KV-cache utilization, block-allocated token capacity, and prefix-cache hit rate
- GPU utilization, temperature, memory, and power draw
- CPU utilization, temperature, and per-core activity
- Memory-pressure and thrashing risk from Linux PSI, swap, reclaim, and fault counters
- InfiniBand/RoCE link state derived from sysfs without requiring `ibstat`

## Requirements

**Control machine**

- macOS or Linux
- Python 3.9 or newer
- OpenSSH client
- Network access to each configured vLLM HTTP endpoint

**Each DGX Spark node**

- Passwordless SSH public-key authentication
- `nvidia-smi`
- Read access to `/proc` and `/sys`
- A running vLLM server exposing `/metrics`

`dgx-top` never needs an SSH password, private key contents, sudo credentials, or an API token in its configuration.

## Quick start

Run from source — no global or tool install needed. `uv` is the only extra
requirement besides Python 3.9+ and an OpenSSH client; `uv sync` installs the
project and its dependencies into a local `.venv`.

```bash
uv sync
uv run dgx-top init      # writes ~/.config/dgx-top/config.toml (mode 0600)
```

Edit the SSH targets and vLLM URLs in `~/.config/dgx-top/config.toml`, then
validate access and launch:

```bash
uv run dgx-top check
uv run dgx-top
```

## Passwordless SSH

If SSH keys are not configured yet:

```bash
ssh-keygen -t ed25519
ssh-copy-id spark@YOUR_SPARK_HOST
ssh spark@YOUR_SPARK_HOST true
```

The final command must complete without asking for a password. `dgx-top check` uses `BatchMode=yes`, so password prompts fail safely instead of hanging the dashboard.

### Using SSH aliases

Aliases keep usernames, hostnames, ports, jump hosts, and identity-file choices in your normal OpenSSH configuration rather than in dgx-top:

```sshconfig
# ~/.ssh/config
Host spark-primary
    HostName spark01.example.com
    User spark
    IdentityFile ~/.ssh/id_ed25519

Host spark-worker
    HostName spark02.example.com
    User spark
    IdentityFile ~/.ssh/id_ed25519
```

Test each alias before continuing:

```bash
ssh spark-primary true
ssh spark-worker true
```

## Configuration

`dgx-top init` creates this structure:

```toml
[app]
poll_interval = 5
history_length = 40

[[nodes]]
label = "spark-1"
ssh_target = "spark-primary"
vllm_url = "http://spark01.example.com:8000"
worker = false

[[nodes]]
label = "spark-2"
ssh_target = "spark-worker"
vllm_url = "http://spark02.example.com:8000"
worker = true
```

| Setting          | Meaning                                                                     |
| ---------------- | --------------------------------------------------------------------------- |
| `label`          | Short unique name shown in the dashboard                                    |
| `ssh_target`     | Any target accepted by `ssh`, such as `user@host` or an SSH alias           |
| `vllm_url`       | vLLM base URL reachable from the control machine; do not include `/metrics` |
| `worker`         | Marks a worker node in a tensor-parallel deployment                         |
| `poll_interval`  | Initial polling interval in seconds, from 1 to 60                           |
| `history_length` | Number of samples retained in memory, from 10 to 1000                       |

The SSH target and vLLM URL are intentionally separate. An SSH alias can resolve through `~/.ssh/config`, while HTTP clients generally cannot use that alias.

To use a different file:

```bash
dgx-top --config /path/to/config.toml check
dgx-top --config /path/to/config.toml
```

Or set `DGX_TOP_CONFIG`.

## Preflight checks

`dgx-top check` tests every node concurrently:

1. The SSH target accepts key-based, non-interactive authentication.
2. `nvidia-smi` is available and `/proc/stat` is readable.
3. The configured vLLM `/metrics` endpoint is reachable and contains vLLM metrics.

No collected telemetry or endpoint response body is written to disk.

## Controls

| Key | Action              |
| --- | ------------------- |
| `+` | Poll faster         |
| `-` | Poll slower         |
| `r` | Refresh immediately |
| `q` | Quit                |

## Troubleshooting

### `Host key verification failed`

Connect once interactively and verify the host fingerprint:

```bash
ssh YOUR_SSH_TARGET true
```

`dgx-top` does not disable SSH host-key verification.

### SSH works interactively but preflight fails

The connection is probably relying on password authentication. Confirm non-interactive access:

```bash
ssh -o BatchMode=yes YOUR_SSH_TARGET true
```

### vLLM metrics fail while SSH passes

The metrics URL is accessed directly from the control machine, not through SSH. Verify it there:

```bash
curl --fail http://YOUR_VLLM_HOST:8000/metrics
```

Check the vLLM bind address, firewall rules, and `vllm_url`.

### A node appears offline

Run `dgx-top check`, then verify `nvidia-smi` and the Linux telemetry files using the same SSH target. A vLLM failure alone does not mark hardware offline.

## Development

```bash
git clone https://github.com/justincasey/dgx-top.git
cd dgx-top
uv sync --extra dev
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv build
```

`uv sync` leaves the `dgx-top` script in `.venv/bin`, which is not on your `PATH`, so always run it
through `uv` while developing:

```bash
uv run dgx-top init   # write ~/.config/dgx-top/config.toml
uv run dgx-top check  # verify SSH, telemetry, and vLLM endpoints
uv run dgx-top        # launch the dashboard
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the contribution workflow.

## Security and privacy

Configuration files are local and ignored by Git. Do not put passwords, private keys, tokens, private hostnames, or private network addresses in issues, logs, screenshots, examples, or commits. See [SECURITY.md](SECURITY.md) for reporting guidance.

## License

[MIT](LICENSE)
