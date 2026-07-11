# Security Policy

## Supported Versions

We actively support security fixes for the following versions of `torchxai-explain`:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |
| < 0.1   | :x:                |

Security fixes are backported to the latest patch release of each supported minor version.

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub Issues.**

If you discover a security vulnerability in `torchxai-explain`, please report it
responsibly using one of the following channels:

### Option 1: GitHub Private Vulnerability Reporting (Preferred)

Use GitHub's built-in private vulnerability reporting:

1. Go to the [Security tab](https://github.com/rigvedrs/torchxai/security) of the repository.
2. Click **"Report a vulnerability"**.
3. Fill in the details and submit.

This keeps the report confidential until a fix is released.

Please include:

- A description of the vulnerability and its potential impact
- Steps to reproduce the issue (or a minimal proof-of-concept)
- The version(s) affected
- Any suggested mitigations (optional)

## Response Timeline

| Milestone                        | Target Timeline   |
| -------------------------------- | ----------------- |
| Initial acknowledgement          | Within 48 hours   |
| Vulnerability confirmed/rejected | Within 7 days     |
| Patch released (if confirmed)    | Within 30 days    |
| Public disclosure                | After patch ships |

We will keep you informed throughout the process. If we need more time to
prepare a fix, we will communicate this and agree on a coordinated disclosure
date.

## Scope

This security policy covers the `torchxai-explain` Python package itself.
Vulnerabilities in upstream dependencies (PyTorch, torchvision, etc.) should
be reported to their respective maintainers.

## Security Best Practices for Users

- **Only load models from trusted sources.** `torchxai-explain` calls
  `torch.load()` internally when loading user-supplied checkpoints; loading
  untrusted `.pt` / `.pth` files can execute arbitrary code (a known PyTorch
  limitation). Use `weights_only=True` where possible.
- Keep your dependencies up-to-date to benefit from upstream security fixes.
- Do not expose raw model outputs or saliency maps from untrusted inputs in
  production without appropriate sanitisation.

## Acknowledgements

We appreciate responsible disclosure and will credit reporters in the release
notes (with permission) for any confirmed vulnerabilities.
