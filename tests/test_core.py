"""Tests for synth-dataset-kit core functionality."""

import json
import tempfile
from pathlib import Path

from synth_dataset_kit.config import SDKConfig, LLMProvider
from synth_dataset_kit.models import Dataset, Example, Message, Role, QualityReport
from synth_dataset_kit.decontamination import Decontaminator
from synth_dataset_kit.exporters import export_jsonl, export_alpaca, export_sharegpt, export_chatml
from synth_dataset_kit.generators.seed_expander import load_seed_file, _parse_example


# ─── CONFIG TESTS ────────────────────────────────────────────────────────────


def test_default_config():
    config = SDKConfig()
    assert config.llm.provider == LLMProvider.OLLAMA
    assert config.generation.num_examples == 100
    assert config.quality.min_score == 7.5
    assert "mmlu" in config.decontamination.benchmarks


def test_config_for_provider():
    config = SDKConfig.default_for_provider("openai")
    assert config.llm.provider == LLMProvider.OPENAI
    assert "openai.com" in config.llm.api_base

    config = SDKConfig.default_for_provider("ollama")
    assert config.llm.provider == LLMProvider.OLLAMA
    assert "11434" in config.llm.api_base


def test_config_yaml_roundtrip():
    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
        path = f.name
    config = SDKConfig.default_for_provider("openai")
    config.to_yaml(path)
    loaded = SDKConfig.from_yaml(path)
    assert loaded.llm.model == config.llm.model
    assert loaded.llm.provider.value == config.llm.provider.value
    Path(path).unlink()


# ─── MODEL TESTS ─────────────────────────────────────────────────────────────


def test_example_creation():
    ex = Example(messages=[
        Message(role=Role.USER, content="Hello"),
        Message(role=Role.ASSISTANT, content="Hi there!"),
    ])
    assert ex.user_message == "Hello"
    assert ex.assistant_message == "Hi there!"
    assert len(ex.id) == 12


def test_dataset_operations():
    ds = Dataset(name="test")
    ds.add(Example(
        messages=[
            Message(role=Role.USER, content="Q1"),
            Message(role=Role.ASSISTANT, content="A1"),
        ],
        quality_score=8.0,
    ))
    ds.add(Example(
        messages=[
            Message(role=Role.USER, content="Q2"),
            Message(role=Role.ASSISTANT, content="A2"),
        ],
        quality_score=4.0,
    ))
    assert ds.size == 2

    filtered = ds.filter_by_quality(7.0)
    assert filtered.size == 1
    assert filtered.examples[0].user_message == "Q1"


def test_dataset_remove_contaminated():
    ds = Dataset(name="test")
    ds.add(Example(
        messages=[
            Message(role=Role.USER, content="Clean"),
            Message(role=Role.ASSISTANT, content="Clean answer"),
        ],
        decontamination_flags=[],
    ))
    ds.add(Example(
        messages=[
            Message(role=Role.USER, content="Contaminated"),
            Message(role=Role.ASSISTANT, content="Bad answer"),
        ],
        decontamination_flags=["mmlu"],
    ))
    clean = ds.remove_contaminated()
    assert clean.size == 1
    assert clean.examples[0].user_message == "Clean"


# ─── PARSER TESTS ────────────────────────────────────────────────────────────


def test_parse_openai_format():
    data = {
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]
    }
    ex = _parse_example(data)
    assert ex is not None
    assert ex.user_message == "Hello"


def test_parse_alpaca_format():
    data = {"instruction": "Translate", "input": "Hello", "output": "Hola"}
    ex = _parse_example(data)
    assert ex is not None
    assert "Hello" in ex.user_message
    assert ex.assistant_message == "Hola"


def test_parse_sharegpt_format():
    data = {
        "conversations": [
            {"from": "human", "value": "What's 2+2?"},
            {"from": "gpt", "value": "4"},
        ]
    }
    ex = _parse_example(data)
    assert ex is not None
    assert ex.user_message == "What's 2+2?"


def test_parse_simple_format():
    data = {"user": "Question", "assistant": "Answer"}
    ex = _parse_example(data)
    assert ex is not None
    assert ex.user_message == "Question"


def test_load_seed_file():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write(json.dumps({"user": "Q1", "assistant": "A1"}) + "\n")
        f.write(json.dumps({"user": "Q2", "assistant": "A2"}) + "\n")
        path = f.name
    examples = load_seed_file(path)
    assert len(examples) == 2
    Path(path).unlink()


# ─── DECONTAMINATION TESTS ──────────────────────────────────────────────────


def test_decontaminator_clean():
    decon = Decontaminator(benchmarks=["mmlu", "gsm8k"], use_benchmark_datasets=False)
    ex = Example(messages=[
        Message(role=Role.USER, content="How do I make pasta?"),
        Message(role=Role.ASSISTANT, content="Boil water, add pasta, cook 8-10 mins."),
    ])
    flags = decon.check_example(ex)
    assert len(flags) == 0


def test_decontaminator_catches_gsm8k():
    decon = Decontaminator(benchmarks=["gsm8k"], use_benchmark_datasets=False)
    ex = Example(messages=[
        Message(role=Role.USER, content="Janet's ducks lay 16 eggs per day. She eats three for breakfast."),
        Message(role=Role.ASSISTANT, content="Let me calculate that."),
    ])
    flags = decon.check_example(ex)
    assert "gsm8k" in flags


def test_decontaminator_catches_humaneval():
    decon = Decontaminator(benchmarks=["humaneval"], use_benchmark_datasets=False)
    ex = Example(messages=[
        Message(role=Role.USER, content="Write a function def has_close_elements that checks if any two numbers are close."),
        Message(role=Role.ASSISTANT, content="Here's the implementation..."),
    ])
    flags = decon.check_example(ex)
    assert "humaneval" in flags


def test_decontaminator_dataset():
    decon = Decontaminator(use_benchmark_datasets=False)
    ds = Dataset(name="test")
    ds.add(Example(messages=[
        Message(role=Role.USER, content="Normal question about cooking"),
        Message(role=Role.ASSISTANT, content="Here's a recipe..."),
    ]))
    ds.add(Example(messages=[
        Message(role=Role.USER, content="Janet's ducks lay 16 eggs per day"),
        Message(role=Role.ASSISTANT, content="The answer is..."),
    ]))
    ds = decon.check_dataset(ds)
    contaminated = [e for e in ds.examples if e.decontamination_flags]
    assert len(contaminated) == 1


# ─── EXPORTER TESTS ─────────────────────────────────────────────────────────


def _make_test_dataset() -> Dataset:
    ds = Dataset(name="test_export")
    for i in range(3):
        ds.add(Example(messages=[
            Message(role=Role.USER, content=f"Question {i}"),
            Message(role=Role.ASSISTANT, content=f"Answer {i}"),
        ]))
    return ds


def test_export_jsonl():
    ds = _make_test_dataset()
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        path = f.name
    export_jsonl(ds, path)
    with open(path) as f:
        lines = f.readlines()
    assert len(lines) == 3
    record = json.loads(lines[0])
    assert "messages" in record
    Path(path).unlink()


def test_export_alpaca():
    ds = _make_test_dataset()
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    export_alpaca(ds, path)
    with open(path) as f:
        records = json.load(f)
    assert len(records) == 3
    assert "instruction" in records[0]
    Path(path).unlink()


def test_export_sharegpt():
    ds = _make_test_dataset()
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        path = f.name
    export_sharegpt(ds, path)
    with open(path) as f:
        lines = f.readlines()
    assert len(lines) == 3
    record = json.loads(lines[0])
    assert "conversations" in record
    Path(path).unlink()


def test_export_chatml():
    ds = _make_test_dataset()
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        path = f.name
    export_chatml(ds, path)
    with open(path) as f:
        lines = f.readlines()
    assert len(lines) == 3
    record = json.loads(lines[0])
    assert "<|im_start|>" in record["text"]
    Path(path).unlink()


# ─── QUALITY REPORT TESTS ───────────────────────────────────────────────────


def test_quality_report():
    report = QualityReport(
        dataset_name="test",
        total_examples=100,
        passed_examples=85,
        failed_examples=15,
        avg_quality_score=7.5,
        diversity_score=0.82,
    )
    assert report.passed_examples + report.failed_examples == report.total_examples


if __name__ == "__main__":
    # Run all tests
    import sys

    test_functions = [v for k, v in globals().items() if k.startswith("test_")]
    passed = 0
    failed = 0
    for test_fn in test_functions:
        try:
            test_fn()
            print(f"  ✓ {test_fn.__name__}")
            passed += 1
        except Exception as e:
            print(f"  ✗ {test_fn.__name__}: {e}")
            failed += 1

    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed > 0 else 0)
