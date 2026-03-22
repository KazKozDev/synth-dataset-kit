# Contributing to synth-dataset-kit

Thanks for your interest in contributing.

## Before you start

- Check [open issues](https://github.com/KazKozDev/synth-dataset-kit/issues) to avoid duplicates
- For large changes, open an issue first to discuss the approach

## Setup

```bash
git clone https://github.com/KazKozDev/synth-dataset-kit.git
cd synth-dataset-kit
pip install -e ".[dev]"
```

## Workflow

1. Fork the repo and create a branch from `main`:
   ```bash
   git checkout -b feat/your-feature
   ```
2. Make your changes
3. Run linter and tests:
   ```bash
   ruff check synth_dataset_kit/
   pytest tests/ -v
   ```
4. Push and open a PR against `main`

## Code style

- `ruff` with `line-length = 100`, target Python 3.10+
- Type hints required for all public functions
- No new dependencies without prior discussion in an issue

## What makes a good PR

- One focused change per PR (no bundling unrelated fixes)
- Tests for new behaviour
- Updated docstrings if public API changed
- Brief description of what changed and why

## Reporting bugs

Use the [bug report template](https://github.com/KazKozDev/synth-dataset-kit/issues/new?template=bug_report.md).

## Security issues

Do **not** open a public issue. See [SECURITY.md](SECURITY.md).
