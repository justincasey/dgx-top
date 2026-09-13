# Security policy

## Reporting a vulnerability

Do not open a public issue for a vulnerability that could expose credentials, private network information, or permit command execution. Use GitHub's private vulnerability reporting feature for this repository when available. If private reporting is unavailable, contact the repository maintainer through the private contact method listed on their GitHub profile.

Include a concise description, affected version or commit, reproduction steps, and impact. Do not include real credentials, private keys, internal addresses, or unredacted telemetry.

## Security model

- dgx-top runs on a trusted control machine under the current user's account.
- It invokes the local OpenSSH client with non-interactive authentication.
- SSH host-key verification remains enabled.
- It executes a fixed read-only telemetry command on configured nodes.
- It performs unauthenticated HTTP GET requests to configured engine endpoints: `/metrics` (vLLM and SGLang), SGLang's `/v1/loads` and `/get_load` load API when metrics are unavailable, and `/v1/models` for served model names.
- It does not store telemetry, credentials, endpoint responses, or history on disk.
- Configuration contains connection targets only and should never contain secret material.

Users are responsible for limiting network exposure of engine metrics endpoints and protecting their SSH private keys using normal operating-system and OpenSSH controls.

## Supported versions

Security fixes are applied to the latest released version and the default branch.
