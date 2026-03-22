"""Optional fine-tuning workflows for in-repo training runs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class TrainingJob:
    dataset_path: str
    base_model: str
    output_dir: str
    trainer: str = "unsloth"
    epochs: int = 1
    learning_rate: float = 2e-4
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 2048
    load_in_4bit: bool = True


def load_jsonl_messages(dataset_path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(dataset_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def render_chat_example(record: dict[str, Any]) -> str:
    messages = record.get("messages", [])
    rendered: list[str] = []
    for message in messages:
        role = str(message.get("role", "user")).strip().lower()
        content = str(message.get("content", "")).strip()
        if not content:
            continue
        rendered.append(f"{role.title()}: {content}")
    return "\n".join(rendered).strip()


def save_training_job(job: TrainingJob) -> list[str]:
    output_dir = Path(job.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "training_job.json"
    json_path.write_text(json.dumps(asdict(job), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    md_path = output_dir / "training_job.md"
    md_lines = [
        "# Training Job",
        "",
        f"- trainer: {job.trainer}",
        f"- base model: {job.base_model}",
        f"- dataset: {job.dataset_path}",
        f"- epochs: {job.epochs}",
        f"- learning rate: {job.learning_rate}",
        f"- batch size: {job.batch_size}",
        f"- grad accumulation: {job.gradient_accumulation_steps}",
        f"- max seq length: {job.max_seq_length}",
    ]
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return [str(json_path), str(md_path)]


def run_training_job(job: TrainingJob) -> dict[str, Any]:
    trainer = job.trainer.strip().lower()
    if trainer != "unsloth":
        raise ValueError(f"Unsupported trainer '{job.trainer}'. Only 'unsloth' is implemented.")
    return run_unsloth_finetune(job)


def run_unsloth_finetune(job: TrainingJob) -> dict[str, Any]:
    try:
        import torch
        from datasets import Dataset as HFDataset
        from transformers import TrainingArguments
        from trl import SFTTrainer
        from unsloth import FastLanguageModel
    except ImportError as exc:
        raise RuntimeError(
            "Unsloth training dependencies are not installed. "
            "Install the training stack first, then rerun `sdk finetune`."
        ) from exc

    records = load_jsonl_messages(job.dataset_path)
    train_texts = [{"text": render_chat_example(record)} for record in records if render_chat_example(record)]
    if not train_texts:
        raise ValueError(f"No trainable chat examples found in {job.dataset_path}")

    output_dir = Path(job.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = output_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=job.base_model,
        max_seq_length=job.max_seq_length,
        load_in_4bit=job.load_in_4bit,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    dataset = HFDataset.from_list(train_texts)
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=job.max_seq_length,
        args=TrainingArguments(
            output_dir=str(artifacts_dir),
            per_device_train_batch_size=job.batch_size,
            gradient_accumulation_steps=job.gradient_accumulation_steps,
            learning_rate=job.learning_rate,
            num_train_epochs=job.epochs,
            logging_steps=1,
            save_strategy="epoch",
            report_to="none",
            fp16=bool(torch.cuda.is_available()),
            bf16=False,
        ),
    )
    train_result = trainer.train()

    final_model_dir = artifacts_dir / "final-model"
    trainer.model.save_pretrained(str(final_model_dir))
    tokenizer.save_pretrained(str(final_model_dir))

    metrics = dict(getattr(train_result, "metrics", {}) or {})
    metrics.update(
        {
            "trainer": job.trainer,
            "base_model": job.base_model,
            "dataset_path": job.dataset_path,
            "output_dir": str(final_model_dir),
            "examples": len(train_texts),
        }
    )
    metrics_path = output_dir / "finetune_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {
        "model_dir": str(final_model_dir),
        "metrics_path": str(metrics_path),
        "examples": len(train_texts),
    }
