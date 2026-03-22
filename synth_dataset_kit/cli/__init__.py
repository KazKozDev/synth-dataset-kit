"""CLI interface for synth-dataset-kit.

Primary flow:
  sdk init          — Configure the tool
  sdk create        — Turn a small seed set into a reviewed dataset bundle
  sdk inspect       — Inspect accepted/rejected artifacts and audit results

Advanced flow:
  sdk go            — Legacy alias for `create --demo`
  sdk generate      — Generate only
  sdk audit         — Audit only
  sdk export        — Export or build publish bundles
  sdk run           — Non-interactive end-to-end pipeline
  sdk health        — Check endpoint connectivity
"""

from __future__ import annotations

import synth_dataset_kit.cli._commands_advanced  # noqa: F401
import synth_dataset_kit.cli._commands_pipeline  # noqa: F401

# Import command modules so their @app.command() decorators register with the app.
import synth_dataset_kit.cli._commands_primary  # noqa: F401
from synth_dataset_kit.cli._app import _default_demo_seed_path, app  # noqa: F401
from synth_dataset_kit.cli._commands_advanced import (  # noqa: F401
    _recommend_benchmark_result,
    _select_benchmark_models,
)

# Re-export internal helpers that tests and other modules depend on.
from synth_dataset_kit.cli._commands_pipeline import (  # noqa: F401
    _artifact_base_name,
    _resolve_artifact_group,
)
from synth_dataset_kit.cli._display import (  # noqa: F401
    _artifact_example_preview,
    _artifact_summary,
    _export_artifact_csv,
    _sort_artifact_examples,
)

__all__ = ["app"]
