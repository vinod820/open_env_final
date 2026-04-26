---
title: SocialEngineerArena
emoji: "🚀"
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
pinned: false
---

# SocialEngineerArena

The SOC dashboard shows a new urgent email.
The sender looks familiar. The request sounds reasonable. The timing is bad.

This is where mistakes happen.

**SocialEngineerArena** is the environment we built to train LLMs for this exact moment: high-pressure enterprise communication where the model must reason, verify, and stay safe instead of reacting fast and wrong.

It is not an attack tool.  
It is a decision-training world.

---

## Judge Quickstart (3-5 Minutes)

- Live environment: [Hugging Face Space](https://huggingface.co/spaces/vinod2005/social-engineer-arena)
- Live learning page (one-click train + live curves/logs): [Live Learning Console](https://huggingface.co/spaces/vinod2005/social-engineer-arena/)
- Previous run evidence page (saved curves + metrics snapshot): [Previous Curves & Results](https://huggingface.co/spaces/vinod2005/social-engineer-arena/)
- Re-runnable training: [Google Colab Notebook](https://colab.research.google.com/drive/1AWQWs_8il-g0JJK7-qw9JcyN_x68u_Er?usp=sharing)
- Mini-blog writeup: `blog.md`
- Command runbook: `RUNBOOK.md`
- Reward/loss plots: `assets/reward_curve.png`, `assets/loss_curve.png`, `assets/grpo_reward_curve.png`

If you only do one thing: open the Space, run a few episodes, then check the plots and logs below.

---

## The Story We Are Solving

Most benchmarks ask, "is this phishing?"

Real teams face a harder question:
"Given messy context, conflicting priorities, and partial information, what should I do next?"

In this environment, each episode is a workplace thread with:

- a role (`attacker` or `defender`)
- policy excerpts and process constraints
- message turns where intent is often ambiguous until late
- delayed consequences

The model must return a structured action:
`verdict`, `explanation`, `cues_found`, `response`, `safety_boundary`.

That turns vague safety claims into behavior we can train and measure.

---

## What We Built

We implemented an OpenEnv-compatible environment and training loop that includes:

- **Environment API**: `reset()`, `step(action)`, `state()`
- **Scenario world**: `social_engineer_arena/data/scenarios.large.json`
- **FastAPI runtime + UI**: deployed to Hugging Face Spaces
- **AI-suggested interaction flow** in the Space for reproducible judging
- **Training pipelines** with TRL SFT and TRL GRPO

This is designed as a proper training system, not a static classification demo.

---

## New Demo Pages in Space

To make judging faster, the Space now includes two dedicated evidence pages:

- **Live Learning Console** (`/insights/live`)
  - one click on **Start Learning** triggers `/train`
  - live status, live training logs, and live loss/reward curves
  - useful for showing training is actually running now

- **Previous Curves & Results** (`/insights/history`)
  - displays saved curve images from `assets/`
  - displays baseline and latest run summaries from `outputs/`
  - useful for showing before/after evidence quickly

Judge flow we recommend:
**Arena behavior demo -> Live Learning run -> Previous Results proof snapshot**.

---

## Why It Matches Hackathon Themes

- **Primary: Theme #3.1 Professional Tasks (World Modeling)**
  - policy-aware enterprise workflow
  - partially observable multi-turn state
  - realistic action-taking under constraints
- **Secondary: Theme #1 Multi-Agent Interactions**
  - attacker/defender incentive dynamics
  - strategic communication behavior across turns

Positioning in one line:
**SocialEngineerArena trains policy-grounded social engineering defense in a realistic professional world, with explicit attacker/defender interaction dynamics.**

---

## How Reward Teaches Behavior

Episodes run across multiple turns, with final reward at episode completion.

Defender scoring weights:

- `0.40` verdict correctness
- `0.25` reasoning quality
- `0.25` cue coverage
- `0.10` calibration and process adherence

Attacker track is sandboxed for simulation only and penalized for unsafe real-world abuse patterns.

The reward is intentionally difficult to game: fast guesses and shallow explanations lose.

---

## What We Trained and How

### Pipeline A: TRL SFT

- Entry: `scripts/train_suggest_model.py`
- Core trainer: `scripts/train_hf_job_sft.py`

### Pipeline B: TRL GRPO (RL)

- Trainer: `scripts/train_trl_grpo.py`
- Compatibility entrypoint: `scripts/train_grpo_placeholder.py`

Colab and local scripts are both included so judges can rerun easily.

---

## Evidence That Training Happened

Artifacts committed in this repo:

- Learning curve: `assets/loss_curve.png`
- Reward curve: `assets/reward_curve.png`
- GRPO step reward curve: `assets/grpo_reward_curve.png`
- SFT log: `outputs/logs/submission_sft_20260426_130833.log`
- GRPO log: `outputs/logs/submission_grpo_20260426_130833.log`
- Baseline metrics: `outputs/evals/baseline_results.json`
- SFT summary: `outputs/submission_sft_20260426_130833/summary.json`
- GRPO summary: `outputs/submission_grpo_20260426_130833/summary_grpo.json`

Observed small-model iteration gain:

- Train split: `0.1007 -> 0.3906` (`+0.2899`)
- Test split: `0.0424 -> 0.3321` (`+0.2897`)

This is the core claim: reward-aware training improves behavior in this environment.

---

## Reproduce in Minutes

```bash
pip install -e ".[dev]"
python scripts/train_suggest_model.py
python scripts/evaluate_baselines.py
python -m social_engineer_arena.server.app
```

Open:

- `http://localhost:8000/arena`
- `http://localhost:8000/web`
- `http://localhost:8000/insights/live`
- `http://localhost:8000/insights/history`

Quick GRPO run:

```bash
PYTHONUTF8=1 PYTHONIOENCODING=utf-8 MODEL_NAME=sshleifer/tiny-gpt2 OUTPUT_DIR=outputs/grpo_quick MAX_STEPS=4 NUM_GENERATIONS=2 MAX_PROMPTS=64 python scripts/train_trl_grpo.py
```

---

## Safety and Scope

- All organizations and domains are fictionalized.
- Unsafe real-world abuse behavior is penalized.
- The objective is defensive capability and robust judgment under pressure.

---

## Closing

Social engineering attacks are language attacks.
So we built a language-first environment where models can learn not just to classify messages, but to reason through pressure, justify actions, and stay aligned with policy.

That is what we trained, and that is what this submission demonstrates.
