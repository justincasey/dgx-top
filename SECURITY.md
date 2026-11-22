# Security policy

## Reporting a vulnerability

Do not open a public issue for a vulnerability that could expose credentials, private network information, or permit command execution. Use GitHub's private vulnerability reporting feature for this repository when available. If private reporting is unavailable, contact the repository maintainer through the private contact method listed on their GitHub profile.

Include a concise description, affected version or commit, reproduction steps, and impact. Do not include real credentials, private keys, internal addresses, or unredacted telemetry.

## Security model

- dgx-top runs on a trusted control machine under the current user's account.
- It invokes the local OpenSSH client with non-interactive authentication.
- SSH host-key verification remains enabled.
- It executes a fixed read-only telemetry command on configured nodes.
- Only on an explicit, confirmed `c` clock-menu action does it change a node: `sudo -n nvidia-smi -lgc 0,<ceiling>` or `sudo -n nvidia-smi -rgc` (non-interactive sudo; it never prompts for or handles a password), plus one marker file under the SSH user's `~/.cache/dgx-top/`. The only value interpolated into that command is the integer ceiling, clamped to 1500–2400 MHz (the validated configuration value, stepped in the menu).
- It performs unauthenticated HTTP GET requests to configured engine endpoints: `/metrics` (vLLM, SGLang and TensorFold), SGLang's `/v1/loads` and `/get_load` load API when metrics are unavailable, and `/v1/models` for served model names.
- It does not store telemetry, credentials, endpoint responses, or history on disk.
- For each configured `[[switches]]` entry it performs HTTP Basic-authenticated REST requests to RouterOS: GETs of `/rest/system/resource`, `/rest/system/health` and `/rest/interface` (and `/rest/system/identity` from `dgx-top check`), plus one `POST /rest/interface/ethernet/monitor` with `once`, a read-only command RouterOS exposes as a POST. It never writes to the switch; give it a user whose group has only the `read`, `api` and `rest-api` policies.
- The switch password is read only from the environment variable named by `password_env`; it never appears in the configuration file, logs, or error text. Basic auth sends it with every request, so over a plain `http` url it crosses the network in cleartext (`dgx-top check` warns). Prefer `https` (RouterOS `www-ssl`), with `ca_file` for a private CA.
- Configuration contains connection targets only and should never contain secret material.

Users are responsible for limiting network exposure of engine metrics endpoints and protecting their SSH private keys using normal operating-system and OpenSSH controls.

## Supported versions

Security fixes are applied to the latest released version and the default branch.
