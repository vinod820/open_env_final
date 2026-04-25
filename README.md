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

SocialEngineerArena is an OpenEnv-compatible RL environment for teaching LLMs to both:

- detect manipulation attempts in enterprise communication threads, and
- generate safe fictional red-team simulations for defender training.

This project targets practical cybersecurity behavior change, not offensive deployment. Attack-mode outputs are safety-constrained and scored down for any real-world abuse patterns.

## Why this environment matters

Real social engineering is rarely one-shot. Attackers exploit urgency, role pressure, and policy exceptions over multiple messages. This environment focuses on:

- multi-turn thread analysis with delayed reward,
- policy-aware decisions under conflicting business pressure,
- calibrated reasoning (reduce false positives on legitimate operational traffic).

## Theme alignment

- **Theme #4 (Self-Improvement):** dual-role red/blue episodes support iterative policy improvement.
- **Theme #3.1 (World Modeling / Professional Tasks):** stateful enterprise context, policy snippets, and conflicting incentives.
- **Theme #1 (Multi-Agent Interaction):** attacker/defender dynamics represented through multi-turn threads.

## Environment design

Each episode is a **2-5 turn scenario** with delayed reward (reward returned at terminal turn).

### Observation

- role, channel, persona, organization
- current turn message
- prior thread history
- user profile
- policy excerpt
- conflicting context signal
- turn index / total turns
- task + rubric metadata

### Action

- `verdict`
- `explanation`
- `cues_found`
- `response`
- `safety_boundary`

### Transition and reward

- `reset()` starts a new scenario with turn index `0`.
- `step(action)` advances one turn.
- Intermediate turns return reward `0.0` (delayed reward setup).
- Terminal turn returns final episode reward:
  - average turn-level score,
  - plus defender consistency bonus for stable, correct verdict behavior.

## Reward rubric

### Defender

- `0.40` verdict correctness
- `0.25` reasoning quality
- `0.25` cue coverage
- `0.10` calibration/process adherence

### Attacker (safe fictional simulation only)

- `0.30` cue coverage
- `0.25` persuasive realism
- `0.20` reasoning quality
- `0.25` safety compliance

## Data and splits

Scenarios are split into:

- `train`
- `test` (unseen split for generalization checks)

Stored in `social_engineer_arena/data/scenarios.json`.

### Public data sources used for large-scale scenario generation

When scaling beyond the handcrafted seed scenarios, this project uses public phishing/safe communication datasets:

- Hugging Face: `cybersectony/PhishingEmailDetectionv2.0`  
  <https://huggingface.co/datasets/cybersectony/PhishingEmailDetectionv2.0>
- Kaggle: The Biggest Spam Ham Phish Email Dataset (250000+)  
  <https://www.kaggle.com/datasets/akshatsharma2/the-biggest-spam-ham-phish-email-dataset-300000>

Large-scenario builder script:

```bash
python scripts/build_large_scenarios.py --output social_engineer_arena/data/scenarios.large.json
```

Optional: include local Kaggle CSV files (downloaded into `data/raw/kaggle`):

```bash
python scripts/build_large_scenarios.py --kaggle-dir data/raw/kaggle --limit-hf 5000 --limit-kaggle 3000
```

## Run locally

```bash
pip install -e ".[dev]"
python -m pytest -q
python scripts/evaluate_baselines.py
python -m social_engineer_arena.server.app
```

Demo UI:

- OpenEnv UI: `http://localhost:8000/web`
- Showcase UI: `http://localhost:8000/arena`

## Evaluation and logging

### Baseline and unseen-split evaluation

```bash
python scripts/evaluate_baselines.py
```

Output:

- `outputs/evals/baseline_results.json`

Includes:

- train split weak vs rubric-aware baseline means
- test split weak vs rubric-aware baseline means
- delta reward on unseen split

### Endpoint rollout with archival logs

```bash
python scripts/run_endpoint_rollout.py --episodes 100 --split test --temperature 0.3 --top-p 0.9
```

Outputs:

- latest:
  - `outputs/endpoint_rollout.jsonl`
  - `outputs/endpoint_rollout.csv`
  - `outputs/endpoint_rollout_reward.png`
- archived:
  - `outputs/runs/<run_id>/endpoint_rollout.jsonl`
  - `outputs/runs/<run_id>/endpoint_rollout.csv`
  - `outputs/runs/<run_id>/endpoint_rollout_reward.png`
  - `outputs/runs/<run_id>/summary.json`

## Training entrypoint

Notebook:

- `notebooks/train_social_engineer_arena_grpo.ipynb`

Use it to connect TRL/Unsloth GRPO training to live environment reward:

- `reset -> model output -> parse -> step -> reward`.

For SFT jobs with larger generated scenarios, set `SCENARIOS_PATH`:

```bash
SCENARIOS_PATH=social_engineer_arena/data/scenarios.large.json python scripts/train_hf_job_sft.py
```

## Hugging Face Space deployment

Target URL:

- `https://huggingface.co/spaces/<your-hf-username>/social-engineer-arena`

Deploy:

```bash
openenv push --repo-id <your-hf-username>/social-engineer-arena
```

## Submission checklist

- [x] OpenEnv manifest (`openenv.yaml`)
- [x] Environment/client/models implementation
- [x] Multi-turn delayed-reward episode design
- [x] Rich composable rubric
- [x] Baseline evaluation script
- [x] Unseen split evaluation output
- [x] Rollout logging + archived run evidence
- [x] Demo UI (`/arena`)
- [ ] Full TRL/Unsloth training run with reward/loss curves
- [ ] Published Space URL in README
- [ ] Mini-blog / <2 min video / slides linked in README

## Safety

- All organizations/domains are fictional.
- Attack-mode content is constrained to harmless training simulation.
- Reward penalizes live links, credential collection, malware/payment instructions, and missing fictional markers.

## Readiness check

```bash
python scripts/check_submission_readiness.py
```
