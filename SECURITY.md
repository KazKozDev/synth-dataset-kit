# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | Yes       |

## Reporting a Vulnerability

**Do not open a public GitHub issue for security vulnerabilities.**

Report vulnerabilities privately by emailing [kazkozdev@gmail.com](mailto:kazkozdev@gmail.com) with:

- A description of the vulnerability
- Steps to reproduce
- Potential impact
- Your suggested fix (optional)

You can expect an acknowledgement within 48 hours and a resolution timeline within 7 days.

## Scope

This project connects to external LLM APIs and processes user-provided data. Areas of particular concern:

- Prompt injection via seed files or domain descriptions passed to the LLM
- API key exposure through config files or logs
- Malicious content in generated datasets (PII, toxic content)

The built-in quality and PII checks (`check_pii: true` in config) provide a baseline defence, but are not a substitute for reviewing generated data before use in production.
