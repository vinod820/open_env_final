"""
Compatibility entrypoint for GRPO training.

Use:
    python scripts/train_grpo_placeholder.py

This wrapper now forwards to the runnable TRL GRPO script:
    scripts/train_trl_grpo.py
"""

from scripts.train_trl_grpo import main


if __name__ == "__main__":
    main()
