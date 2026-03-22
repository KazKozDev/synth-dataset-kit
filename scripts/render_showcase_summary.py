#!/usr/bin/env python3
"""Render a compact showcase summary from run artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from synth_dataset_kit.showcase import render_showcase_summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Render a showcase summary from run artifacts.")
    parser.add_argument("run_summary", help="Path to run_summary.json")
    parser.add_argument("--output", "-o", help="Write Markdown to this path")
    args = parser.parse_args()

    output_path = Path(args.output) if args.output else None
    written_path = render_showcase_summary(Path(args.run_summary), output_path=output_path)
    if not args.output:
        print(written_path.read_text(encoding="utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
