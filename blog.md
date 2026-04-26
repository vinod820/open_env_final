# SocialEngineerArena Outputs Explained (Beginner Friendly)

This guide explains the project outputs in simple language.
If you are new to ML, you should still be able to understand what happened after running training or evaluation.

---

## Big Picture

When you run this project, it creates different kinds of outputs:

- **Model files** (the trained brain)
- **Metrics files** (numbers that tell if it improved)
- **Rollout logs** (what happened in each episode)
- **Plots/images** (quick visual summary)

Think of it like exam results:

- model files = your final answer sheet
- metrics = your score report
- logs = step-by-step rough work

---

## 1) Hugging Face Model Repo Outputs

Example repo: `vinod2005/social-engineer-arena-suggest`

Common files you will see there:

- `model.safetensors`  
  The actual trained model weights. This is the most important file.

- `config.json`  
  Model settings (architecture info, label mappings, etc.).

- `tokenizer.json` and `tokenizer_config.json`  
  Rules for converting text into tokens and back.

- `generation_config.json`  
  Default text generation settings.

- `training_args.bin`  
  Training configuration saved by Transformers.

- `README.md` (model card)  
  Human-readable info about the model.

### How to interpret

If these files exist in the model repo, your model push was successful.

---

## 2) Training Console Outputs (what logs mean)

During training, you may see lines like:

- `loss`
- `grad_norm`
- `learning_rate`
- `mean_token_accuracy`
- `train_runtime`
- `train_steps_per_second`

### Simple meaning

- **loss**: Lower is usually better.  
  It means "how wrong the model currently is."

- **mean_token_accuracy**: Higher is better.  
  Roughly how often token predictions are correct.

- **learning_rate**: Step size for learning.  
  Not a score, just a control value.

- **train_runtime**: Total training time.

- **train_steps_per_second**: Training speed.

### Example from your recent run

- Train steps: `35/35` (training finished)
- Final train loss: around `1.032`
- Mean token accuracy rose to around `0.9288`
- Model upload reached `100%`

This means training itself completed and the model artifacts were uploaded.

---

## 3) Local Evaluation Outputs

When running baseline evaluation (`scripts/evaluate_baselines.py`):

- `outputs/evals/baseline_results.json`

### What this contains

- scores for weak baseline vs rubric-aware baseline
- split-wise results (`train`, `test`)
- deltas showing improvement

### How to interpret

If rubric-aware score is higher than weak baseline, your reward logic is helping model behavior.

---

## 4) Endpoint Rollout Outputs

When running endpoint rollouts (`scripts/run_endpoint_rollout.py`), output files include:

- `outputs/endpoint_rollout.jsonl`
- `outputs/endpoint_rollout.csv`
- `outputs/endpoint_rollout_reward.png`
- plus archived copies in `outputs/runs/<run_id>/...`

### What each file is for

- **JSONL**: One episode per line (detailed, machine-friendly).
- **CSV**: Same idea in table format (Excel-friendly).
- **PNG reward plot**: Visual trend of reward over episodes.
- **summary.json**: short summary of the run.

### How to interpret quickly

- more positive rewards = generally better behavior
- high parse success + stable rewards = healthier endpoint behavior

---

## 5) API Outputs from the App

The app endpoints return structured JSON:

- `POST /reset` -> new scenario observation
- `POST /step` -> next observation + reward + done
- `POST /suggest` -> model-generated action suggestion
- `POST /train` -> starts background training
- `GET /train/status` -> current training status/logs

### What to check first

- `status` in `/train/status`:
  - `running` = still training
  - `completed` = finished successfully
  - `failed` = ended with error

- `exit_code`:
  - `0` usually means success
  - non-zero means failure

---

## 6) Why a job can show "timeout" even after success

This can happen when:

1. training and upload finished, but
2. extra post-training steps run too long (like reward evaluation), and
3. job hits timeout before clean exit

So always verify model repo files on Hugging Face.
If `model.safetensors` and tokenizer files are present, the core model push likely succeeded.

---

## 7) Practical "Is my run good?" checklist

Use this quick checklist:

- [ ] Training reached final step count (`N/N`)
- [ ] Final loss lower than early loss
- [ ] Model files exist on HF repo
- [ ] Endpoint rollout rewards are not all near zero
- [ ] No major parsing failures in rollout logs

If most are true, your run is healthy.

---

## Final Note

You do not need perfect metrics in one run.
For this project, the key is:

- safe behavior,
- stable improvement,
- and reproducible outputs.

That is already a strong result.
