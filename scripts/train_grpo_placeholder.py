"""
Minimal GRPO/TRL training scaffold.

This file is intentionally lightweight for pre-onsite work. Use it after receiving
Hugging Face compute credits by replacing MODEL_NAME and connecting the reward_fn
to live SocialEngineerArena episodes.
"""

from __future__ import annotations

from social_engineer_arena.models import ArenaAction
from social_engineer_arena.server.environment import SocialEngineerArenaEnvironment

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


def format_prompt(obs) -> str:
    return f"""You are playing SocialEngineerArena.
Role: {obs.role}
Persona: {obs.persona}
Organization: {obs.organization}
Context: {obs.thread_context}
Message: {obs.incoming_message}
Task: {obs.task}

Return JSON with verdict, explanation, cues_found, response, safety_boundary.
"""


def reward_fn(completions: list[str]) -> list[float]:
    env = SocialEngineerArenaEnvironment()
    rewards: list[float] = []
    for completion in completions:
        obs = env.reset()
        try:
            action = ArenaAction.model_validate_json(completion)
        except Exception:
            action = ArenaAction(explanation=completion)
        _, reward, _ = env.step(action)
        rewards.append(reward)
    return rewards


def main() -> None:
    print("Install social-engineer-arena[train], then wire this reward_fn into TRL GRPOTrainer.")
    print(f"Suggested starter model: {MODEL_NAME}")


if __name__ == "__main__":
    main()
