#!/usr/bin/env python
# /// script
# dependencies = [
#   "trl>=0.18.0",
#   "transformers>=4.44.0",
#   "datasets>=2.20.0",
#   "accelerate>=0.33.0",
#   "peft>=0.12.0",
#   "matplotlib>=3.8.0",
#   "torch>=2.2.0",
#   "social-engineer-arena @ git+https://github.com/vinod820/open_env_final.git"
# ]
# ///
from __future__ import annotations

"""
Train a suggest-action model for SocialEngineerArena.

This wrapper sets sane defaults for suggestion quality and then runs the
existing SFT trainer script (`train_hf_job_sft.py`).
"""

import os
import runpy
import tempfile
from pathlib import Path
from urllib.request import urlretrieve


def main() -> None:
    os.environ.setdefault("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")
    os.environ.setdefault("OUTPUT_REPO", "vinod2005/social-engineer-arena-suggest")
    os.environ.setdefault("OUTPUT_DIR", "outputs/social_engineer_arena_suggest")
    os.environ.setdefault("DATA_MULTIPLIER", "32")
    os.environ.setdefault("MAX_STEPS", "180")
    os.environ.setdefault("LEARNING_RATE", "2e-5")
    os.environ.setdefault("GRAD_ACCUM_STEPS", "8")
    os.environ.setdefault("MAX_LENGTH", "256")
    os.environ.setdefault("EVAL_STRATEGY", "steps")
    os.environ.setdefault("EVAL_STEPS", "20")
    os.environ.setdefault("SAVE_STEPS", "40")
    os.environ.setdefault("MAX_EVAL_SCENARIOS", "150")
    os.environ.setdefault("GEN_MAX_NEW_TOKENS", "180")
    os.environ.setdefault("PUSH_TO_HUB", "1")

    trainer_script = Path(__file__).with_name("train_hf_job_sft.py")
    if not trainer_script.exists():
        trainer_url = os.getenv(
            "TRAINER_SCRIPT_URL",
            "https://raw.githubusercontent.com/vinod820/open_env_final/main/scripts/train_hf_job_sft.py",
        )
        tmp_script = Path(tempfile.gettempdir()) / "train_hf_job_sft.py"
        print(f"Trainer script not found locally. Downloading from: {trainer_url}")
        urlretrieve(trainer_url, tmp_script)
        trainer_script = tmp_script
    runpy.run_path(str(trainer_script), run_name="__main__")


if __name__ == "__main__":
    main()
