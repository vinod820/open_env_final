# Hackathon Runbook (Top-15 Mode)

This runbook gives a repeatable path for fast iteration + proof artifacts.

## 1) Quick Proof Run (local or Colab)

- SFT (TRL): `scripts/train_suggest_model.py`
- RL (TRL GRPO): `scripts/train_trl_grpo.py`
- Baseline eval: `scripts/evaluate_baselines.py`
- Reward curve: `scripts/make_reward_plot.py`

Expected proof artifacts:

- `outputs/*/loss_curve.png`
- `assets/reward_curve.png`
- `outputs/evals/baseline_results.json`
- training/eval logs in `outputs/logs/`

## 2) Recommended small-model loop

Use small model first:

- `MODEL_NAME=sshleifer/tiny-gpt2` (plumbing/proof)
- then `MODEL_NAME=Qwen/Qwen2.5-0.5B-Instruct` (stronger quality)

Tune in this order:

1. Reward function quality
2. Prompt/output format stability
3. Training steps/data multiplier

## 3) Commands

### SFT quick loop

```bash
PYTHONUTF8=1 PYTHONIOENCODING=utf-8 MODEL_NAME=sshleifer/tiny-gpt2 OUTPUT_DIR=outputs/sft_quick MAX_STEPS=12 DATA_MULTIPLIER=1 PUSH_TO_HUB=0 python scripts/train_suggest_model.py
```

### GRPO quick loop

```bash
PYTHONUTF8=1 PYTHONIOENCODING=utf-8 MODEL_NAME=sshleifer/tiny-gpt2 OUTPUT_DIR=outputs/grpo_quick MAX_STEPS=4 NUM_GENERATIONS=2 MAX_PROMPTS=64 python scripts/train_trl_grpo.py
```

### Eval + reward curve

```bash
python scripts/evaluate_baselines.py
python scripts/make_reward_plot.py
```

## 4) Submission checklist

- [ ] Space URL in `README.md`
- [ ] Colab notebook link in `README.md`
- [ ] Mini-blog or <2min video link in `README.md`
- [ ] Loss curve + reward curve committed
- [ ] Real logs committed or linked
- [ ] Before/after reward evidence shown

