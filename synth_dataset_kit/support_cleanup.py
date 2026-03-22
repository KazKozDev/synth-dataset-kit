"""Post-processing helpers for support-style dataset cleanup."""

from __future__ import annotations

import json
import re
from pathlib import Path


_ACTION_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(
            r"\bI(?:'ll| will)\s+"
            r"(open|issue|arrange|check|investigate|review|update|process|submit|trigger|"
            r"escalate|flag|create|send|confirm|cancel|refund|ship|unlock|reset)\b",
            re.IGNORECASE,
        ),
        r"Support can \1",
    ),
    (
        re.compile(
            r"\bI(?:'m| am)\s+"
            r"(opening|issuing|arranging|checking|investigating|reviewing|updating|processing|"
            r"submitting|triggering|escalating|flagging|creating|sending|confirming)\b",
            re.IGNORECASE,
        ),
        r"The next step is for support to \1",
    ),
    (
        re.compile(
            r"\bI can\s+"
            r"(open|issue|arrange|check|investigate|review|update|process|submit|trigger|"
            r"escalate|flag|create|send|confirm|cancel|refund|ship|unlock|reset|replace)\b",
            re.IGNORECASE,
        ),
        r"Support can \1",
    ),
    (
        re.compile(
            r"\bWe can\s+"
            r"(open|issue|arrange|check|investigate|review|update|process|submit|trigger|"
            r"escalate|flag|create|send|confirm|cancel|refund|ship|unlock|reset|replace)\b",
            re.IGNORECASE,
        ),
        r"Support can \1",
    ),
]

_TEXT_REPLACEMENTS: list[tuple[str, str]] = [
    ("right away", "if the request is eligible"),
    ("right now", "next"),
    ("I’ve checked", "Once support checks"),
    ("I've checked", "Once support checks"),
    ("I checked", "Once support checks"),
    ("I can see", "Support should be able to verify"),
    ("we don't make you wait", "the usual next step is"),
]

_AWKWARD_REPLACEMENTS: list[tuple[str, str]] = [
    ("lose access as the next step", "lose access immediately"),
    ("effective as the next step on our side", "effective once the change is applied on our side"),
    ("effective once the change is applied on our side", "effective once the change is applied"),
    ("typically typically", "typically"),
]

_TIMELINE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(r"\bwithin (\d+\s*(?:hours?|business days?|days?))\b", re.IGNORECASE),
        r"typically within \1",
    ),
    (
        re.compile(r"\bin (\d+\s*(?:hours?|business days?|days?))\b", re.IGNORECASE),
        r"typically in \1",
    ),
]

_RISK_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bI(?:'ll| will)\b", re.IGNORECASE),
    re.compile(r"\bI can\b", re.IGNORECASE),
    re.compile(r"\bSupport can\b", re.IGNORECASE),
    re.compile(r"\bwe(?:'ll| can)\b", re.IGNORECASE),
    re.compile(r"\bescalate\b", re.IGNORECASE),
    re.compile(r"\brefund\b", re.IGNORECASE),
    re.compile(r"\bship\b", re.IGNORECASE),
    re.compile(r"\bon our side\b", re.IGNORECASE),
    re.compile(r"\bwithin \d+", re.IGNORECASE),
    re.compile(r"\btypically within\b", re.IGNORECASE),
]


def soften_support_answer(text: str) -> str:
    """Reduce overconfident support phrasing without changing the core advice."""
    updated = text
    for pattern, replacement in _ACTION_PATTERNS:
        updated = pattern.sub(replacement, updated)
    for source, target in _TEXT_REPLACEMENTS:
        updated = updated.replace(source, target)
    for pattern, replacement in _TIMELINE_PATTERNS:
        updated = pattern.sub(replacement, updated)
    for source, target in _AWKWARD_REPLACEMENTS:
        updated = updated.replace(source, target)
    return updated


def targeted_support_answer_review(text: str) -> str:
    """Apply a stronger, more conservative rewrite for the riskiest answers."""
    updated = soften_support_answer(text)
    targeted_patterns: list[tuple[re.Pattern[str], str]] = [
        (re.compile(r"\bLet me\b", re.IGNORECASE), "The next step is to"),
        (re.compile(r"\bI(?:'m| am) going to\b", re.IGNORECASE), "The next step is to"),
        (re.compile(r"\bI(?:'ll| will) check\b", re.IGNORECASE), "Support should check"),
        (re.compile(r"\bI(?:'ll| will) pull\b", re.IGNORECASE), "Support should review"),
        (
            re.compile(r"\bI(?:'ll| will) investigate\b", re.IGNORECASE),
            "Support should investigate",
        ),
        (re.compile(r"\bI(?:'ll| will) verify\b", re.IGNORECASE), "Support should verify"),
        (re.compile(r"\bI can try to\b", re.IGNORECASE), "Support can try to"),
        (re.compile(r"\bI can help\b", re.IGNORECASE), "Support should help"),
        (re.compile(r"\bI can initiate\b", re.IGNORECASE), "Support can initiate"),
        (re.compile(r"\bI can request\b", re.IGNORECASE), "Support can request"),
        (re.compile(r"\bI can confirm\b", re.IGNORECASE), "Support should confirm"),
        (re.compile(r"\bI can prioritize\b", re.IGNORECASE), "Support can prioritize"),
        (re.compile(r"\bI’ll\b", re.IGNORECASE), "Support will"),
        (re.compile(r"\bWe’ll\b", re.IGNORECASE), "Support will"),
        (re.compile(r"\bwe’ll\b", re.IGNORECASE), "support will"),
        (re.compile(r"\bI’ll flag\b", re.IGNORECASE), "Support can flag"),
        (re.compile(r"\bI’ll escalate\b", re.IGNORECASE), "Support can escalate"),
        (re.compile(r"\bI’ll arrange\b", re.IGNORECASE), "Support can arrange"),
        (re.compile(r"\bI’ll issue\b", re.IGNORECASE), "Support can issue"),
        (re.compile(r"\bI’ll process\b", re.IGNORECASE), "Support can process"),
        (re.compile(r"\bI’ll create\b", re.IGNORECASE), "Support can create"),
        (re.compile(r"\bI’ll send\b", re.IGNORECASE), "Support can send"),
        (re.compile(r"\bI’ll email\b", re.IGNORECASE), "Support can email"),
        (re.compile(r"\bI’ll refund\b", re.IGNORECASE), "Support can refund"),
        (re.compile(r"\bI’ll ship\b", re.IGNORECASE), "Support can ship"),
        (re.compile(r"\bI’ll open\b", re.IGNORECASE), "Support can open"),
        (re.compile(r"\bI’ll update\b", re.IGNORECASE), "Support can update"),
        (re.compile(r"\bI’ll cancel\b", re.IGNORECASE), "Support can cancel"),
    ]
    for pattern, replacement in targeted_patterns:
        updated = pattern.sub(replacement, updated)
    final_cleanup: list[tuple[str, str]] = [
        ("Support will investigate", "Support should investigate"),
        ("Support will verify", "Support should verify"),
        ("Support will check", "Support should check"),
        ("Support will review", "Support should review"),
        ("support will", "support can"),
        ("on our side", "in the support workflow"),
        ("we can’t directly", "support can’t directly"),
        ("we can usually", "support can usually"),
        ("we can still", "support can still"),
        ("we can either", "support can either"),
        ("we can also", "support can also"),
    ]
    for source, target in final_cleanup:
        updated = updated.replace(source, target)
    return updated


def _risk_score(text: str) -> int:
    return sum(len(pattern.findall(text)) for pattern in _RISK_PATTERNS)


def sanitize_support_jsonl(path: str | Path) -> int:
    """Rewrite generated records in a JSONL dataset with safer support phrasing.

    Returns the number of modified records.
    """
    file_path = Path(path)
    modified = 0
    rewritten: list[str] = []

    for line in file_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        metadata = record.get("metadata") or {}
        if metadata.get("source") == "generated" and record.get("messages"):
            last_message = record["messages"][-1]
            content = str(last_message.get("content", ""))
            softened = soften_support_answer(content)
            if softened != content:
                last_message["content"] = softened
                metadata["support_style_sanitized"] = True
                modified += 1
        rewritten.append(json.dumps(record, ensure_ascii=False))

    file_path.write_text("\n".join(rewritten) + "\n", encoding="utf-8")
    return modified


def curate_top_risky_generated(path: str | Path, top_n: int = 20) -> int:
    """Apply a stronger rewrite to the riskiest generated records only."""
    file_path = Path(path)
    records: list[dict] = []
    risky: list[tuple[int, int]] = []

    for index, line in enumerate(file_path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        record = json.loads(line)
        records.append(record)
        metadata = record.get("metadata") or {}
        if metadata.get("source") != "generated" or not record.get("messages"):
            continue
        if metadata.get("manual_style_reviewed"):
            continue
        score = _risk_score(str(record["messages"][-1].get("content", "")))
        if score:
            risky.append((score, len(records) - 1))

    risky.sort(reverse=True)
    modified = 0
    for _, record_index in risky[:top_n]:
        record = records[record_index]
        metadata = record.get("metadata") or {}
        content = str(record["messages"][-1].get("content", ""))
        rewritten = targeted_support_answer_review(content)
        if rewritten != content:
            record["messages"][-1]["content"] = rewritten
            metadata["manual_style_reviewed"] = True
            record["metadata"] = metadata
            modified += 1

    file_path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )
    return modified
