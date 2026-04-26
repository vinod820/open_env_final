#!/usr/bin/env python
# /// script
# dependencies = [
#   "trl>=1.2.0",
#   "transformers>=4.44.0",
#   "datasets>=2.20.0",
#   "accelerate>=0.33.0",
#   "torch>=2.2.0",
# ]
# ///
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer

from social_engineer_arena.server.environment import SocialEngineerArenaEnvironment


MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "outputs/grpo_quick")
MAX_STEPS = int(os.getenv("MAX_STEPS", "10"))
NUM_GENERATIONS = int(os.getenv("NUM_GENERATIONS", "2"))
MAX_PROMPTS = int(os.getenv("MAX_PROMPTS", "200"))
SEED = int(os.getenv("SEED", "42"))
SCENARIOS_PATH = os.getenv("SCENARIOS_PATH", "").strip()


def _extract_json_block(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    raw = text.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s*```$", "", raw)
    try:
        return json.loads(raw)
    except Exception:
        pass
    m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def _prompt_from_row(row: dict[str, Any]) -> str:
    return (
        "You are playing SocialEngineerArena.\n"
        f"Role: {row['role']}\n"
        f"Incoming Message: {row['incoming_message']}\n"
        f"Policy Excerpt: {row['policy_excerpt']}\n"
        f"Expected Label: {row['label']}\n"
        "Return ONLY one JSON object with keys: "
        "verdict, explanation, cues_found, response, safety_boundary.\n"
        "No markdown.\n"
    )


def build_dataset() -> Dataset:
    kwargs: dict[str, Any] = {}
    if SCENARIOS_PATH:
        kwargs["scenarios_path"] = Path(SCENARIOS_PATH)
    env = SocialEngineerArenaEnvironment(**kwargs)
    rows: list[dict[str, Any]] = []
    for scenario in env.scenarios:
        role = scenario.get("role", "defender")
        policy_excerpt = scenario.get("policy_excerpt", "")
        for turn in scenario.get("turns", []):
            row = {
                "role": role,
                "incoming_message": turn.get("incoming_message", ""),
                "policy_excerpt": policy_excerpt,
                "label": turn.get("label", scenario.get("label", "unknown")),
                "target_cues": turn.get("target_cues", []),
            }
            row["prompt"] = _prompt_from_row(row)
            rows.append(row)
            if len(rows) >= MAX_PROMPTS:
                return Dataset.from_list(rows)
    return Dataset.from_list(rows)


def reward_valid_json(completions: list[str], **_: Any) -> list[float]:
    scores: list[float] = []
    for completion in completions:
        obj = _extract_json_block(completion)
        if obj is None:
            scores.append(0.0)
            continue
        required = {"verdict", "explanation", "cues_found", "response", "safety_boundary"}
        scores.append(1.0 if required.issubset(set(obj.keys())) else 0.3)
    return scores


def reward_label_match(completions: list[str], label: list[str], role: list[str], **_: Any) -> list[float]:
    scores: list[float] = []
    for completion, gold_label, row_role in zip(completions, label, role, strict=False):
        obj = _extract_json_block(completion)
        if obj is None:
            scores.append(0.0)
            continue
        verdict = str(obj.get("verdict", "unknown")).lower().strip()
        if row_role == "attacker":
            # attacker rows are simulation generation tasks; reward harmless "safe" framing
            scores.append(1.0 if verdict in {"safe", "unknown"} else 0.2)
        else:
            scores.append(1.0 if verdict == str(gold_label).lower().strip() else 0.0)
    return scores


def reward_cue_overlap(completions: list[str], target_cues: list[list[str]], **_: Any) -> list[float]:
    scores: list[float] = []
    for completion, cues in zip(completions, target_cues, strict=False):
        obj = _extract_json_block(completion)
        if obj is None:
            scores.append(0.0)
            continue
        found = obj.get("cues_found", [])
        if not isinstance(found, list):
            scores.append(0.0)
            continue
        found_l = {str(x).strip().lower() for x in found}
        cues_l = {str(x).strip().lower() for x in cues}
        if not cues_l:
            scores.append(0.5)
            continue
        overlap = len(found_l.intersection(cues_l)) / max(1, len(cues_l))
        scores.append(float(overlap))
    return scores


def main() -> None:
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    ds = build_dataset()
    if len(ds) == 0:
        raise RuntimeError("Dataset is empty. Check scenarios path/config.")

    has_cuda = torch.cuda.is_available()
    config = GRPOConfig(
        output_dir=str(out_dir),
        max_steps=MAX_STEPS,
        per_device_train_batch_size=max(2, NUM_GENERATIONS),
        gradient_accumulation_steps=1,
        num_generations=NUM_GENERATIONS,
        max_completion_length=180,
        logging_steps=1,
        save_steps=max(2, MAX_STEPS),
        report_to="none",
        learning_rate=1e-6,
        bf16=False,
        fp16=has_cuda,
        use_cpu=not has_cuda,
        gradient_checkpointing=has_cuda,
        seed=SEED,
    )

    trainer = GRPOTrainer(
        model=MODEL_NAME,
        args=config,
        train_dataset=ds,
        reward_funcs=[reward_valid_json, reward_label_match, reward_cue_overlap],
    )
    trainer.train()
    trainer.save_model(str(out_dir / "final"))
    summary = {
        "model_name": MODEL_NAME,
        "output_dir": str(out_dir),
        "steps": MAX_STEPS,
        "num_generations": NUM_GENERATIONS,
        "dataset_rows": len(ds),
        "use_cpu": not has_cuda,
    }
    (out_dir / "summary_grpo.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

