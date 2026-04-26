from __future__ import annotations

from social_engineer_arena.models import ArenaAction
from social_engineer_arena.server.environment import SocialEngineerArenaEnvironment


def choose_action(obs) -> ArenaAction:
    # Baseline heuristic keeps output deterministic and parser-friendly.
    if obs.role == "defender":
        return ArenaAction(
            verdict="unknown",
            explanation="Need stronger evidence and sender verification before classification.",
            cues_found=["insufficient-context"],
            response="Please verify via trusted internal channel before any action.",
            safety_boundary="Defensive analysis only.",
        )
    return ArenaAction(
        verdict="safe",
        explanation="Fictional simulation with non-operational content only.",
        cues_found=["fictional-simulation"],
        response="This is a harmless tabletop simulation message with no live targets.",
        safety_boundary="Fictional training only; no real-world abuse guidance.",
    )


def main() -> None:
    env = SocialEngineerArenaEnvironment()
    obs = env.reset()
    print(f"[START] task=social-engineer-arena scenario_id={obs.scenario_id}", flush=True)

    step_idx = 0
    done = False
    while not done and step_idx < max(1, obs.total_turns):
        action = choose_action(obs)
        obs, reward, done = env.step(action)
        step_idx += 1
        print(
            f"[STEP] step={step_idx} reward={reward:.4f} done={str(done).lower()} verdict={action.verdict}",
            flush=True,
        )

    final_score = obs.last_reward if done else 0.0
    print(
        f"[END] task=social-engineer-arena score={final_score:.4f} steps={step_idx}",
        flush=True,
    )


if __name__ == "__main__":
    main()
