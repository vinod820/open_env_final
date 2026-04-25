from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def status_line(name: str, state: str, detail: str) -> str:
    return f"[{state:<7}] {name}: {detail}"


def exists(rel: str) -> bool:
    return (ROOT / rel).exists()


def main() -> None:
    checks: list[tuple[str, str, str]] = []

    # Core OpenEnv compliance.
    checks.append(
        (
            "OpenEnv manifest",
            "PASS" if exists("openenv.yaml") else "PENDING",
            "openenv.yaml present" if exists("openenv.yaml") else "openenv.yaml missing",
        )
    )
    checks.append(
        (
            "Environment implementation",
            "PASS" if exists("social_engineer_arena/server/environment.py") else "PENDING",
            "Environment class present" if exists("social_engineer_arena/server/environment.py") else "Missing environment.py",
        )
    )

    # Training/script requirements.
    has_notebook = exists("notebooks/train_social_engineer_arena_grpo.ipynb")
    has_rollout = exists("scripts/run_endpoint_rollout.py")
    checks.append(
        (
            "Training notebook/script",
            "PASS" if has_notebook and has_rollout else "PARTIAL" if (has_notebook or has_rollout) else "PENDING",
            "Notebook + rollout script present" if has_notebook and has_rollout else "Incomplete training assets",
        )
    )

    # Evidence artifacts.
    has_baseline = exists("outputs/evals/baseline_results.json")
    has_rollout_csv = exists("outputs/endpoint_rollout.csv")
    has_rollout_jsonl = exists("outputs/endpoint_rollout.jsonl")
    has_any_run_archive = (ROOT / "outputs/runs").exists() and any((ROOT / "outputs/runs").iterdir())
    evidence_state = "PASS" if (has_baseline and has_rollout_csv and has_rollout_jsonl) else "PARTIAL" if (has_baseline or has_rollout_csv or has_rollout_jsonl) else "PENDING"
    checks.append(
        (
            "Training/eval evidence",
            evidence_state,
            "Baseline + rollout logs found" if evidence_state == "PASS" else "Need baseline and rollout outputs",
        )
    )
    checks.append(
        (
            "Archived run logs",
            "PASS" if has_any_run_archive else "PENDING",
            "outputs/runs/* exists" if has_any_run_archive else "No archived rollout runs found",
        )
    )

    # External submission items (manual).
    checks.append(("HF Space URL in README", "PENDING", "Set final Space URL before submission"))
    checks.append(("Mini-blog/video/slides link", "PENDING", "Add at least one link in README"))
    checks.append(("Loss + reward training curves", "PENDING", "Add real training curves after onsite run"))

    print("SocialEngineerArena Hackathon Readiness")
    print("=" * 42)
    for name, state, detail in checks:
        print(status_line(name, state, detail))

    score_map = {"PASS": 1.0, "PARTIAL": 0.5, "PENDING": 0.0}
    numeric = sum(score_map[s] for _, s, _ in checks)
    print("-" * 42)
    print(f"Checklist completion score: {numeric:.1f} / {len(checks):.1f}")
    print("Note: External items (Space URL, blog/video, final training curves) require manual completion.")


if __name__ == "__main__":
    main()
