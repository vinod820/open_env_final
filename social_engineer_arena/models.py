from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


Role = Literal["defender", "attacker"]
Verdict = Literal["safe", "phishing", "pretexting", "unknown"]


class ArenaAction(BaseModel):
    """Agent response for one scenario."""

    verdict: Verdict = Field(
        default="unknown",
        description="Defender mode classification. Use safe, phishing, or pretexting.",
    )
    explanation: str = Field(
        default="",
        description="Short rationale. Reward favors concrete evidence from the message.",
    )
    cues_found: list[str] = Field(
        default_factory=list,
        description="Specific manipulation or legitimacy cues identified by the agent.",
    )
    response: str = Field(
        default="",
        description="Attacker mode fictional lure, or defender mode user-facing reply.",
    )
    safety_boundary: str = Field(
        default="",
        description="How the answer avoids real credential theft, malware, impersonation, or harm.",
    )


class RewardBreakdown(BaseModel):
    total: float = 0.0
    verdict: float = 0.0
    reasoning: float = 0.0
    cue_coverage: float = 0.0
    calibration: float = 0.0
    persuasion: float = 0.0
    safety: float = 0.0
    notes: list[str] = Field(default_factory=list)


class ArenaObservation(BaseModel):
    episode_id: str
    role: Role
    scenario_id: str
    channel: str
    persona: str
    organization: str
    thread_context: str
    incoming_message: str
    thread_history: list[str] = Field(default_factory=list)
    user_profile: str = ""
    policy_excerpt: str = ""
    conflicting_context: str = ""
    turn_index: int = 0
    total_turns: int = 1
    task: str
    allowed_verdicts: list[Verdict]
    rubric: dict[str, Any]
    last_reward: float | None = None
    reward_breakdown: RewardBreakdown | None = None
    done: bool = False


class ArenaState(BaseModel):
    episode_id: str = ""
    step_count: int = 0
    scenario_index: int = 0
    role: Role = "defender"
    scenario_id: str = ""
    turn_index: int = 0
    total_turns: int = 1
    cumulative_turn_reward: float = 0.0
    turn_rewards: list[float] = Field(default_factory=list)
    verdict_history: list[Verdict] = Field(default_factory=list)
    done: bool = False
    last_reward: float | None = None
    reward_breakdown: RewardBreakdown | None = None
