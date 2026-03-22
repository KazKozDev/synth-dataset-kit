"""Publishing helpers for dataset bundles."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def resolve_hf_token(token: str | None = None) -> str | None:
    """Resolve a Hugging Face token from args or environment."""
    value = (token or "").strip()
    if value:
        return value
    for env_name in ["HF_TOKEN", "HUGGINGFACEHUB_API_TOKEN"]:
        candidate = os.getenv(env_name, "").strip()
        if candidate:
            return candidate
    return None


def build_publish_manifest(
    repo_id: str,
    bundle_dir: str,
    private: bool,
    pushed: bool,
    uploaded_files: int = 0,
    dataset_url: str | None = None,
) -> dict[str, Any]:
    """Create a compact manifest describing the publish attempt."""
    return {
        "repo_id": repo_id,
        "repo_type": "dataset",
        "bundle_dir": bundle_dir,
        "private": private,
        "pushed": pushed,
        "uploaded_files": uploaded_files,
        "dataset_url": dataset_url or f"https://huggingface.co/datasets/{repo_id}",
    }


def write_publish_manifest(bundle_dir: str, manifest: dict[str, Any]) -> str:
    """Persist publish metadata next to the exported bundle."""
    path = Path(bundle_dir) / "publish_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return str(path)


def publish_huggingface_bundle(
    bundle_dir: str,
    repo_id: str,
    token: str | None = None,
    private: bool = False,
    commit_message: str = "Publish dataset bundle from synth-dataset-kit",
    exist_ok: bool = True,
) -> dict[str, Any]:
    """Upload a prepared dataset bundle to the Hugging Face Hub."""
    resolved_token = resolve_hf_token(token)
    if not resolved_token:
        raise RuntimeError("Missing Hugging Face token. Set `HF_TOKEN` or pass `--token`.")

    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError(
            "Publishing requires `huggingface_hub`. Install it with "
            "`pip install synth-dataset-kit[publish]`."
        ) from exc

    bundle_path = Path(bundle_dir)
    if not bundle_path.exists():
        raise FileNotFoundError(f"Bundle directory not found: {bundle_dir}")

    api = HfApi(token=resolved_token)
    api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=private,
        exist_ok=exist_ok,
    )
    api.upload_folder(
        folder_path=str(bundle_path),
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=commit_message,
    )

    uploaded_files = sum(1 for path in bundle_path.rglob("*") if path.is_file())
    return build_publish_manifest(
        repo_id=repo_id,
        bundle_dir=str(bundle_path),
        private=private,
        pushed=True,
        uploaded_files=uploaded_files,
    )
