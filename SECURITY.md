# Security Policy

EMAP is exploratory research software. It has not been independently audited,
and this policy does not create a bug bounty, service-level agreement, or
guarantee of payment.

## Supported version

Security fixes target the current `main` branch. Please reproduce an issue
against the latest commit before reporting it.

## Report privately

Use GitHub's private vulnerability reporting form:

https://github.com/noah-ing/EMAP/security/advisories/new

If the form is unavailable, open a public issue asking for a private reporting
channel, but do not include exploit details, credentials, sensitive data, or
working generated-code payloads in that issue.

Include a minimal reproduction, affected commit and component, expected and
observed behavior, impact, preconditions, and any suggested remediation.

The generated-code evaluator is explicitly a best-effort experiment harness,
not a security sandbox. Python object-model escapes and platform-specific
resource-limit gaps are therefore documented limitations. Run it only in an
unprivileged external sandbox with no secrets or sensitive network/filesystem
access. Reports remain useful when the implementation violates a documented
fail-closed behavior, silently falls back to in-process execution, exposes a
credential, or introduces an unsafe default outside those stated limits.

Do not test against systems, accounts, models, or data you do not own or have
written permission to assess. Do not exfiltrate data, degrade a service, or
retain sensitive material beyond what is necessary to demonstrate the issue.

Please allow a reasonable remediation window before public disclosure.
