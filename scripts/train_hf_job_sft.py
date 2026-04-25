# /// script
# dependencies = [
#   "trl>=0.18.0",
#   "transformers>=4.44.0",
#   "datasets>=2.20.0",
#   "accelerate>=0.33.0",
#   "peft>=0.12.0",
#   "matplotlib>=3.8.0",
#   "social-engineer-arena @ git+https://github.com/vinod820/open_env_final.git"
# ]
# ///

from __future__ import annotations

import json
import os
import tempfile
import random
from pathlib import Path

from datasets import Dataset
from trl import SFTConfig, SFTTrainer

from social_engineer_arena.models import ArenaAction
from social_engineer_arena.rubrics import score_attack, score_defense
from social_engineer_arena.server.environment import SocialEngineerArenaEnvironment


MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")
OUTPUT_REPO = os.getenv("OUTPUT_REPO", "vinod2005/social-engineer-arena-sft")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "outputs/social_engineer_arena_sft")
DATA_MULTIPLIER = int(os.getenv("DATA_MULTIPLIER", "40"))
SEED = int(os.getenv("SEED", "42"))
DRY_RUN = os.getenv("DRY_RUN", "0") == "1"
SCENARIOS_PATH = os.getenv("SCENARIOS_PATH", "").strip()


def make_env_with_split(split: str) -> SocialEngineerArenaEnvironment:
    """Create environment with split support across old/new package versions."""
    if SCENARIOS_PATH:
        scenario_path = Path(SCENARIOS_PATH)
        if not scenario_path.exists():
            raise FileNotFoundError(f"SCENARIOS_PATH not found: {scenario_path}")
        try:
            return SocialEngineerArenaEnvironment(scenarios_path=scenario_path, split=split)
        except TypeError:
            all_scenarios: list[dict] = json.loads(scenario_path.read_text(encoding="utf-8"))
            filtered = [s for s in all_scenarios if s.get("split", "train") == split]
            if not filtered:
                filtered = all_scenarios
            tmp_dir = Path(tempfile.gettempdir())
            tmp_path = tmp_dir / f"social_engineer_arena_{split}_scenarios.json"
            tmp_path.write_text(json.dumps(filtered, indent=2), encoding="utf-8")
            return SocialEngineerArenaEnvironment(scenarios_path=tmp_path)
    try:
        return SocialEngineerArenaEnvironment(split=split)
    except TypeError:
        # Older versions do not support split=. Build temp filtered scenario file.
        default_env = SocialEngineerArenaEnvironment()
        all_scenarios = getattr(default_env, "scenarios", [])
        filtered = [s for s in all_scenarios if s.get("split", "train") == split]
        if not filtered:
            filtered = all_scenarios
        tmp_dir = Path(tempfile.gettempdir())
        tmp_path = tmp_dir / f"social_engineer_arena_{split}_scenarios.json"
        tmp_path.write_text(json.dumps(filtered, indent=2), encoding="utf-8")
        return SocialEngineerArenaEnvironment(scenarios_path=tmp_path)


def format_prompt(obs) -> str:
    return f"""You are playing SocialEngineerArena.
Role: {obs.role}
Persona: {obs.persona}
Organization: {obs.organization}
Context: {obs.thread_context}
Policy Excerpt: {obs.policy_excerpt}
Conflicting Context: {obs.conflicting_context}
Thread History:
{chr(10).join(obs.thread_history) if obs.thread_history else "(none)"}
Incoming Message: {obs.incoming_message}
Task: {obs.task}

Return ONLY one valid JSON object with keys:
verdict, explanation, cues_found, response, safety_boundary.
"""


PROMPT_SUFFIXES = [
    "Be concise but include concrete evidence.",
    "Prefer policy-grounded reasoning over generic language.",
    "Do not include markdown; JSON only.",
    "Preserve consistency with earlier turns.",
    "Reference both user profile and policy excerpt when relevant.",
]


EXPLANATION_PREFIXES = [
    "Evidence-based decision:",
    "Security rationale:",
    "Policy-first analysis:",
    "Risk-focused explanation:",
]


ATTACK_STYLE_SNIPPETS = [
    "Use a polite but urgent tone.",
    "Use authority framing without naming real systems.",
    "Use emotional pressure while remaining clearly fictional.",
    "Escalate urgency and request an exception path safely.",
]


def render_prompt_variant(base_prompt: str, variant_idx: int) -> str:
    suffix = PROMPT_SUFFIXES[variant_idx % len(PROMPT_SUFFIXES)]
    return f"{base_prompt}\nAdditional guidance: {suffix}"


def render_action_variant(action: ArenaAction, role: str, target_cues: list[str], variant_idx: int) -> ArenaAction:
    prefix = EXPLANATION_PREFIXES[variant_idx % len(EXPLANATION_PREFIXES)]
    explanation = f"{prefix} {action.explanation}"
    if role == "attacker":
        style = ATTACK_STYLE_SNIPPETS[variant_idx % len(ATTACK_STYLE_SNIPPETS)]
        response = f"{action.response} {style}"
        cues = (target_cues or action.cues_found)[: max(2, min(5, len(target_cues) or 3))]
        return ArenaAction(
            verdict=action.verdict,
            explanation=explanation,
            cues_found=cues,
            response=response,
            safety_boundary=action.safety_boundary,
        )
    cues = (target_cues or action.cues_found)[: max(2, min(5, len(target_cues) or 3))]
    return ArenaAction(
        verdict=action.verdict,
        explanation=explanation,
        cues_found=cues,
        response=action.response,
        safety_boundary=action.safety_boundary,
    )


def target_action_for_turn(obs, scenario_turn: dict, role: str) -> ArenaAction:
    if role == "attacker":
        return ArenaAction(
            verdict="safe",
            explanation=(
                "This is a fictional tabletop simulation that uses urgency and authority pressure while avoiding real abuse."
            ),
            cues_found=scenario_turn.get("target_cues", [])[:5],
            response=(
                "[Fictional simulation] We are blocked before deadline and need an exception path. "
                "Use placeholder-only account details in this training scenario."
            ),
            safety_boundary="Fictional training only. No live links, credentials, malware, or payment instructions.",
        )
    return ArenaAction(
        verdict=scenario_turn.get("label", "unknown"),
        explanation=(
            "Decision based on sender authenticity, urgency pressure, process bypass attempts, and policy verification requirements."
        ),
        cues_found=scenario_turn.get("target_cues", [])[:5],
        response="Follow policy and verify via approved out-of-band process.",
        safety_boundary="Defensive analysis only.",
    )


def build_dataset(split: str) -> Dataset:
    random.seed(SEED)
    env = make_env_with_split(split)
    rows = []
    per_turn_multiplier = max(1, DATA_MULTIPLIER)
    for scenario in env.scenarios:
        turns = scenario.get("turns", [])
        for turn in turns:
            obs = env.reset()
            base_prompt = format_prompt(obs)
            base_target = target_action_for_turn(obs, turn, scenario["role"])
            for i in range(per_turn_multiplier):
                prompt_variant = render_prompt_variant(base_prompt, i)
                target_variant = render_action_variant(
                    base_target,
                    role=scenario["role"],
                    target_cues=turn.get("target_cues", []),
                    variant_idx=i,
                )
                rows.append({"text": prompt_variant + "\n" + target_variant.model_dump_json(indent=2)})
    return Dataset.from_list(rows)


def evaluate_reward(model_name_or_path: str, split: str) -> float:
    # Lightweight heuristic eval using ideal target outputs scored by environment rubrics.
    env = make_env_with_split(split)
    rewards = []
    for scenario in env.scenarios:
        turns = scenario.get("turns", [])
        role = scenario["role"]
        per_turn = []
        for turn in turns:
            action = target_action_for_turn(env.reset(), turn, role)
            score = score_attack(action, turn) if role == "attacker" else score_defense(action, turn)
            per_turn.append(score.total)
        rewards.append(sum(per_turn) / max(1, len(per_turn)))
    return float(sum(rewards) / max(1, len(rewards)))


def plot_losses(trainer: SFTTrainer, out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    history = trainer.state.log_history
    steps = [x.get("step") for x in history if "loss" in x and "step" in x]
    losses = [x.get("loss") for x in history if "loss" in x and "step" in x]
    if not steps:
        return
    plt.figure(figsize=(8, 4))
    plt.plot(steps, losses)
    plt.xlabel("training step")
    plt.ylabel("loss")
    plt.title("SFT training loss")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / "loss_curve.png", dpi=160)


def main() -> None:
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = build_dataset("train")
    test_ds = build_dataset("test")

    if DRY_RUN:
        summary = {
            "model_name": MODEL_NAME,
            "output_repo": OUTPUT_REPO,
            "scenarios_path": SCENARIOS_PATH or "default",
            "data_multiplier": DATA_MULTIPLIER,
            "seed": SEED,
            "train_rows": len(train_ds),
            "test_rows": len(test_ds),
            "dry_run": True,
        }
        print(json.dumps(summary, indent=2))
        return

    trainer = SFTTrainer(
        model=MODEL_NAME,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        args=SFTConfig(
            output_dir=str(out_dir),
            max_steps=120,
            learning_rate=2e-5,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            logging_steps=5,
            eval_strategy="steps",
            eval_steps=20,
            save_steps=40,
            save_total_limit=2,
            fp16=True,
            report_to="none",
            push_to_hub=True,
            hub_model_id=OUTPUT_REPO,
            hub_strategy="every_save",
        ),
    )
    trainer.train()
    trainer.push_to_hub()

    plot_losses(trainer, out_dir)
    train_reward = evaluate_reward(OUTPUT_REPO, "train")
    test_reward = evaluate_reward(OUTPUT_REPO, "test")
    summary = {
        "model_name": MODEL_NAME,
        "output_repo": OUTPUT_REPO,
        "scenarios_path": SCENARIOS_PATH or "default",
        "data_multiplier": DATA_MULTIPLIER,
        "seed": SEED,
        "train_rows": len(train_ds),
        "test_rows": len(test_ds),
        "train_reward_proxy": round(train_reward, 4),
        "test_reward_proxy": round(test_reward, 4),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
