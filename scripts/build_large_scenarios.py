#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import random
import re
from pathlib import Path
from typing import Any

from datasets import load_dataset


DEFAULT_OUTPUT = Path("social_engineer_arena/data/scenarios.large.json")

DEFENDER_PERSONAS = [
    "Security analyst reviewing suspicious communications",
    "IT support engineer triaging user-reported alerts",
    "Finance operations specialist handling sensitive requests",
    "Product manager coordinating urgent cross-team launches",
]

ORG_NAMES = [
    "Northstar Robotics",
    "FleetForge",
    "Cedar Health",
    "PixelBank",
    "HarborCloud",
    "MetroLedger",
]

POLICY_EXCERPTS = [
    "Never share credentials, OTPs, or verification codes in email/chat/SMS.",
    "Urgent account or payroll changes require independent out-of-band verification.",
    "Do not open untrusted links or attachments from unknown or spoofed senders.",
    "Escalations must still follow identity checks and documented approval flows.",
]

CONFLICTING_CONTEXTS = [
    "Leadership is pushing for speed due to quarter-end deadlines.",
    "The team is handling an active incident and inbox pressure is high.",
    "Launch-day blockers increase urgency and social pressure.",
    "Support queues are overloaded, creating temptation to skip verification.",
]


def sanitize_text(value: Any) -> str:
    text = str(value or "").strip()
    text = re.sub(r"\s+", " ", text)
    return text


def guess_label(content: str, raw_label: Any) -> str:
    label_text = str(raw_label).lower().strip()
    content_l = content.lower()
    if "pretext" in label_text:
        return "pretexting"
    if label_text in {"phishing", "1", "true", "malicious"}:
        return "phishing"
    if label_text in {"safe", "legitimate", "ham", "0", "false", "benign"}:
        return "safe"

    suspicious_tokens = [
        "verify your account",
        "urgent",
        "password",
        "otp",
        "bank account",
        "gift card",
        "wire transfer",
        "click here",
        "reset now",
    ]
    return "phishing" if any(tok in content_l for tok in suspicious_tokens) else "safe"


def infer_cues(content: str, label: str) -> tuple[list[str], list[str]]:
    content_l = content.lower()
    cues: list[str] = []
    legit: list[str] = []

    if any(x in content_l for x in ["urgent", "immediately", "asap", "final warning"]):
        cues.append("urgency pressure")
    if any(x in content_l for x in ["verify your account", "login", "password", "otp", "mfa"]):
        cues.append("credential solicitation")
    if any(x in content_l for x in ["confidential", "secret", "do not tell", "private"]):
        cues.append("secrecy pressure")
    if any(x in content_l for x in ["click", "http://", "https://", "bit.ly", "tinyurl"]):
        cues.append("link-based action request")
    if any(x in content_l for x in ["gift card", "wire", "bank", "payment"]):
        cues.append("financial manipulation")

    if label == "safe":
        legit.extend(["routine business tone", "no credential solicitation"])
    else:
        legit.extend(["message appears manipulative", "requires verification"])

    return cues[:5], legit[:3]


def make_turn(message: str, label: str) -> dict[str, Any]:
    target_cues, legitimate_cues = infer_cues(message, label)
    return {
        "incoming_message": message,
        "target_cues": target_cues or (["social engineering cue"] if label != "safe" else ["standard business context"]),
        "legitimate_cues": legitimate_cues,
        "label": label,
    }


def make_scenario(idx: int, message: str, label: str, split: str) -> dict[str, Any]:
    role = "defender"
    persona = DEFENDER_PERSONAS[idx % len(DEFENDER_PERSONAS)]
    org = ORG_NAMES[idx % len(ORG_NAMES)]
    return {
        "id": f"large-{split}-{idx:06d}",
        "split": split,
        "role": role,
        "channel": "email",
        "persona": persona,
        "organization": org,
        "thread_context": "This sample is auto-generated from public phishing/safe communication datasets for defender training.",
        "user_profile": "User is expected to follow policy under time pressure and avoid risky shortcuts.",
        "policy_excerpt": POLICY_EXCERPTS[idx % len(POLICY_EXCERPTS)],
        "conflicting_context": CONFLICTING_CONTEXTS[idx % len(CONFLICTING_CONTEXTS)],
        "label": label,
        "difficulty": 0.75 if label in {"phishing", "pretexting"} else 0.55,
        "turns": [make_turn(message, label)],
    }


def rows_from_hf(limit: int | None, seed: int) -> list[tuple[str, str]]:
    ds = load_dataset("cybersectony/PhishingEmailDetectionv2.0")
    train_rows = list(ds["train"])
    random.Random(seed).shuffle(train_rows)
    if limit:
        train_rows = train_rows[:limit]

    pairs: list[tuple[str, str]] = []
    for row in train_rows:
        content = sanitize_text(row.get("content", ""))
        if len(content) < 20:
            continue
        # Dataset labels: 0 legit email, 1 phishing email, 2 legit url, 3 phishing url
        raw = row.get("labels")
        mapped = "safe" if str(raw) in {"0", "2"} else "phishing"
        pairs.append((content, mapped))
    return pairs


def rows_from_kaggle_dir(kaggle_dir: Path, limit: int | None) -> list[tuple[str, str]]:
    if not kaggle_dir.exists():
        return []

    csv_files = sorted(kaggle_dir.rglob("*.csv"))
    pairs: list[tuple[str, str]] = []
    for file in csv_files:
        with file.open("r", encoding="utf-8", errors="ignore", newline="") as f:
            reader = csv.DictReader(f)
            fields = [x.lower() for x in (reader.fieldnames or [])]
            text_key = next((k for k in ["content", "text", "message", "body", "email"] if k in fields), None)
            label_key = next((k for k in ["label", "labels", "type", "class", "category"] if k in fields), None)
            if not text_key:
                continue
            for row in reader:
                message = sanitize_text(row.get(text_key, ""))
                if len(message) < 20:
                    continue
                label = guess_label(message, row.get(label_key, ""))
                pairs.append((message, label))
                if limit and len(pairs) >= limit:
                    return pairs
    return pairs


def split_pairs(pairs: list[tuple[str, str]], test_ratio: float, seed: int) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    rnd = random.Random(seed)
    shuffled = pairs[:]
    rnd.shuffle(shuffled)
    test_count = max(1, int(len(shuffled) * test_ratio))
    test = shuffled[:test_count]
    train = shuffled[test_count:]
    return train, test


def main() -> None:
    parser = argparse.ArgumentParser(description="Build larger SocialEngineerArena scenarios from public datasets.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output scenarios JSON path.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--limit-hf", type=int, default=5000, help="Max rows from Hugging Face dataset.")
    parser.add_argument("--limit-kaggle", type=int, default=3000, help="Max rows from local Kaggle CSVs.")
    parser.add_argument("--test-ratio", type=float, default=0.2, help="Unseen split ratio.")
    parser.add_argument(
        "--kaggle-dir",
        type=Path,
        default=Path("data/raw/kaggle"),
        help="Directory containing downloaded Kaggle CSV files.",
    )
    args = parser.parse_args()

    hf_rows = rows_from_hf(limit=args.limit_hf, seed=args.seed)
    kaggle_rows = rows_from_kaggle_dir(args.kaggle_dir, limit=args.limit_kaggle)
    all_rows = hf_rows + kaggle_rows
    if not all_rows:
        raise RuntimeError("No rows found. Check internet access for HF dataset or local Kaggle CSV directory.")

    train_pairs, test_pairs = split_pairs(all_rows, test_ratio=args.test_ratio, seed=args.seed)
    scenarios: list[dict[str, Any]] = []

    for idx, (message, label) in enumerate(train_pairs):
        scenarios.append(make_scenario(idx=idx, message=message, label=label, split="train"))
    base = len(scenarios)
    for offset, (message, label) in enumerate(test_pairs):
        scenarios.append(make_scenario(idx=base + offset, message=message, label=label, split="test"))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(scenarios, indent=2), encoding="utf-8")

    summary = {
        "output": str(args.output),
        "total_scenarios": len(scenarios),
        "train_scenarios": len(train_pairs),
        "test_scenarios": len(test_pairs),
        "hf_rows_used": len(hf_rows),
        "kaggle_rows_used": len(kaggle_rows),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
