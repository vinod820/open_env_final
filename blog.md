# Training LLMs to Resist Social Engineering: Building SocialEngineerArena with OpenEnv

Social engineering is rarely a one-shot trick. It is a multi-turn pressure game: urgency, authority framing, policy bypass requests, and context manipulation over time.

Our goal with **SocialEngineerArena** is simple:
train language models to make safer, policy-grounded decisions in those situations, while still supporting fictional red-team simulation for defender training.

## Live demo

- Hugging Face Space: [https://huggingface.co/spaces/vinod2005/social-engineer-arena](https://huggingface.co/spaces/vinod2005/social-engineer-arena)
- Training notebook (Colab): [https://colab.research.google.com/drive/1AWQWs_8il-g0JJK7-qw9JcyN_x68u_Er?usp=sharing](https://colab.research.google.com/drive/1AWQWs_8il-g0JJK7-qw9JcyN_x68u_Er?usp=sharing)

## How to use the Space

1. Open the Space URL and wait for the app to become ready.
2. Click **Start Episode** to load a scenario.
3. Review the message context (role, policy excerpt, thread history).
4. Use **Suggest Action** to get model-generated JSON action.
5. Edit/submit via **Submit Action** and observe reward + breakdown.
6. Repeat across multiple episodes to inspect consistency.
7. Use API routes (`/reset`, `/step`, `/suggest`, `/train`, `/train/status`) for automation.

## Visual snapshots

### Reward improvement curve

![Reward curve](assets/reward_curve.png)

### Training loss curve (local submission run)

![Loss curve](assets/loss_curve.png)

### GRPO run curve (step reward)

![GRPO reward curve](assets/grpo_reward_curve.png)

## Why this environment

Most LLM safety tasks are static classification problems. Real enterprise communication is not.

This environment models:

- multi-turn message threads
- attacker and defender roles
- delayed rewards at episode end
- policy context and conflicting business pressure

That makes it a better training target for real-world decision behavior.

## What we built

We built an OpenEnv-compatible environment with:

- role-aware observations (`attacker`/`defender`)
- structured JSON actions (`verdict`, `explanation`, `cues_found`, `response`, `safety_boundary`)
- composable reward logic for defensive correctness, cue coverage, reasoning quality, and safety compliance
- a FastAPI runtime + Space deployment

## Training pipelines

We implemented two training paths:

1. **TRL SFT pipeline**  
   `scripts/train_suggest_model.py` -> `scripts/train_hf_job_sft.py`

2. **TRL GRPO pipeline (RL)**  
   `scripts/train_trl_grpo.py`  
   (also wired through `scripts/train_grpo_placeholder.py` for compatibility)

A re-runnable Colab is included and linked from README.

## Evidence from real runs

### Baseline reward improvement

From `outputs/evals/baseline_results.json`:

- **Train split:** `0.1007 -> 0.3906` (`+0.2899`)
- **Test split:** `0.0424 -> 0.3321` (`+0.2897`)

This indicates meaningful gain from rubric-aware behavior over weak baseline behavior.

### Artifact files

- Loss curve: `assets/loss_curve.png`
- Reward curve: `assets/reward_curve.png`
- GRPO step reward curve: `assets/grpo_reward_curve.png`
- SFT logs: `outputs/logs/submission_sft_20260426_130833.log`
- GRPO logs: `outputs/logs/submission_grpo_20260426_130833.log`
- SFT summary: `outputs/submission_sft_20260426_130833/summary.json`
- GRPO summary: `outputs/submission_grpo_20260426_130833/summary_grpo.json`

## What worked best

The strongest practical strategy was:

- start with small models for fast loops
- improve reward signal quality first
- run many short iterations
- scale up model/compute only after reward and behavior are stable

This gave better progress than trying to force a very large model too early.

## Quick reproducibility commands

```bash
python scripts/train_suggest_model.py
python scripts/evaluate_baselines.py
python scripts/make_reward_plot.py
python scripts/train_trl_grpo.py
```

## Safety approach

The environment is explicitly safety-constrained:

- all scenarios are fictionalized
- attack-mode outputs are penalized for real abuse patterns
- rewards favor process adherence and defensive verification

## What we would improve next

- stronger structured-output guarantees in GRPO completions
- richer long-horizon scenarios with harder delayed reward credit assignment
- additional “before vs after” qualitative episode walkthroughs for judges

---

SocialEngineerArena is not just a benchmark. It is a training loop for safer decision behavior under realistic communication pressure.
