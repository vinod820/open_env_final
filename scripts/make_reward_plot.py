from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RESULTS_JSON = ROOT / "outputs" / "evals" / "baseline_results.json"
PNG = ROOT / "assets" / "reward_curve.png"


def load_means() -> tuple[float, float, float, float]:
    if not RESULTS_JSON.exists():
        raise FileNotFoundError(
            f"Missing {RESULTS_JSON}. Run `python scripts/evaluate_baselines.py` first."
        )
    payload = json.loads(RESULTS_JSON.read_text(encoding="utf-8"))
    deltas = {x["split"]: x for x in payload.get("delta_metrics", [])}
    train = deltas.get("train")
    test = deltas.get("test")
    if not train or not test:
        raise ValueError("baseline_results.json missing train/test delta_metrics entries.")
    return (
        float(train["weak_mean_reward"]),
        float(train["rubric_aware_mean_reward"]),
        float(test["weak_mean_reward"]),
        float(test["rubric_aware_mean_reward"]),
    )


def main() -> None:
    train_weak, train_rubric, test_weak, test_rubric = load_means()
    labels = ["train", "test"]
    weak_vals = [train_weak, test_weak]
    rubric_vals = [train_rubric, test_rubric]

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(labels, weak_vals, marker="o", linewidth=2.2, label="Weak baseline")
    ax.plot(labels, rubric_vals, marker="o", linewidth=2.2, label="Rubric-aware baseline")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Mean reward")
    ax.set_xlabel("Data split")
    ax.set_title("Baseline comparison on SocialEngineerArena")
    ax.grid(alpha=0.25)
    ax.legend()
    for i, y in enumerate(weak_vals):
        ax.annotate(f"{y:.3f}", (labels[i], y), textcoords="offset points", xytext=(0, -14), ha="center")
    for i, y in enumerate(rubric_vals):
        ax.annotate(f"{y:.3f}", (labels[i], y), textcoords="offset points", xytext=(0, 8), ha="center")

    PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(PNG, dpi=160)
    print(f"Wrote {PNG}")


if __name__ == "__main__":
    main()
