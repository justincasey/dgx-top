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
- CPU utilization, temperature, frequency, and per-core activity
- Memory-pressure and thrashing risk from Linux PSI, swap, reclaim, and fault counters
- InfiniBand/RoCE link state and throughput (RX/TX rates and wire utilization) derived from sysfs without requiring `ibstat`

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
theme = "dgx-dark"  # or any name from `dgx-top themes`

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
| `theme`          | Color theme name; see [Themes](#themes)                                    |

The SSH target and vLLM URL are intentionally separate. An SSH alias can resolve through `~/.ssh/config`, while HTTP clients generally cannot use that alias.

To use a different file:

```bash
dgx-top --config /path/to/config.toml check
dgx-top --config /path/to/config.toml
```
Or set `DGX_TOP_CONFIG`.

## Themes

The dashboard is themed with Textual's native theme system. Set the theme in the
configuration file:

```toml
[app]
theme = "tokyo-night"
```

or override it on the command line:

```bash
dgx-top --theme tokyo-night-storm
```

Every Textual built-in theme is supported out of the box — including the full
Tokyo Night family — plus the classic `dgx-dark` default:

- **Dark:** `dgx-dark` (default), `tokyo-night`, `tokyo-night-storm`, `nord`,
  `gruvbox`, `dracula`, `monokai`, `catppuccin-mocha`, `catppuccin-macchiato`,
  `catppuccin-frappe`, `rose-pine`, `rose-pine-moon`, `solarized-dark`,
  `atom-one-dark`, `textual-dark`, `flexoki`, `ansi-dark`
- **Light:** `tokyo-night-light`, `catppuccin-latte`, `solarized-light`,
  `rose-pine-dawn`, `atom-one-light`, `textual-light`, `ansi-light`

Run `dgx-top themes` for the complete list. An unknown theme name is rejected
with the available options when the configuration is loaded.

### Switching themes while running

Press `t` in the dashboard to open a fuzzy theme picker. Highlighting an entry
previews it live and `enter` keeps it; `escape` restores the previous theme. The
switch is session-only — set `theme` in the configuration file to make it stick.

## Preflight checks

`dgx-top check` tests every node concurrently:

1. The SSH target accepts key-based, non-interactive authentication.
2. `nvidia-smi` is available and `/proc/stat` is readable.
3. The configured vLLM `/metrics` endpoint is reachable and contains vLLM metrics.

No collected telemetry or endpoint response body is written to disk.

## Layout and small terminals

The dashboard is fluid in both axes. Width picks the column count:

| Width       | Layout                                             |
| ----------- | -------------------------------------------------- |
| `>= 90`     | Throughput and both nodes side by side (3 columns)  |
| `46`–`89`   | Throughput spans two columns, nodes below           |
| `< 46`      | Everything stacked in one column                    |

Height picks the **density**. dgx-top divides the rows left over after the title
by the number of grid rows the column tier needs, then takes the loosest layout
that fits — so it degrades one step at a time instead of snapping between
extremes, and one column simply needs three times the height for the same
density.

| Rows per tile | Density   | Look                                                     |
| ------------- | --------- | -------------------------------------------------------- |
| `>= 14`       | `roomy`   | `GPU`/`MEMORY`/`CPU` get their own header lines           |
| `10`–`13`     | `dense`   | Those headers become `GPU`/`MEM`/`CPU` row prefixes       |
| `< 10`        | `compact` | One-letter prefixes, meters folded onto their value rows  |

Within a density nothing is left stranded: the sparklines and meters are elastic,
so a viewport between two breakpoints grows the waveforms and thickens the bars
instead of leaving dead space. Growth is bounded — charts stop at eight rows,
meters at two, tiles at 16 — so a tall terminal stays a dashboard rather than
turning into wallpaper. Below the compact minimum the grid stops stretching and
the screen scrolls rather than clipping anything.

Three columns of `dense` tiles fit 12 terminal rows (about 180 pixels); a
one-column `compact` stack fits a 320x320-pixel viewport, roughly 40x21 cells:

```
dgx-top ⚡DUAL 5s  +- t r q
P 0·3600·7200 ▂▃▅▂▇▃▅▁▂▃▅▂▇▃▅▁▂▃▅▂▇▃▅
G 0·1200·2400 ▂▃▅▂▇▃▅▁▂▃▅▂▇▃▅▁▂▃▅▂▇▃▅
2r  1w  h 45%  3:1
1.2M/3.8M 32%
████░░░░░░░░░░░░░░░░  ▂▃▅▂▇▃▅▁▂▃▅▂▇▃▅▁▂▃▅
spark-head Qwen3.6-2…
G 73% 64°C ██████████████████░░░░░░░░░░░
M 62G/120G 52% s1.0G ██████████░░░░░░░░░
C 48% 51°C █████████████░░░░░░░░░░░░░░░░
■■■■■■■■■■■■■■■■■■■■
```

No metric is dropped at any size; compact relocates them:

- Section labels shorten to one letter: `G` GPU, `M` memory, `C` CPU,
  `P` prompt throughput, `G` generation throughput.
- Node value rows share their line with their meter.
- `min avg max` labels collapse to a fixed `min·avg·max` order.
- Throughput has priority: its two sparklines keep full height. The
  `THROUGHPUT` row disappears and its ratio joins the KV request row.
- Swap shortens to `s1.0G`, and the CPU core grid drops its inter-core spacing
  while still showing every core.

## Controls

| Key | Action              |
| --- | ------------------- |
| `+` | Poll faster         |
| `-` | Poll slower         |
| `r` | Refresh immediately |
| `t` | Open the fuzzy theme picker (see [Themes](#switching-themes-while-running)) |
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
