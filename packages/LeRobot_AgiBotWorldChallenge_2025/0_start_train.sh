#!/bin/bash

python -m lerobot.scripts.train \
  --policy.path=lerobot/pi0 \
  --dataset.repo_id=/home/v-yuboguo/LeRobot_AgiBotWorldChallenge_2025/datasets \
  --output_dir=outputs/train/pi0_base_10tasks_n_action_steps_20_test \
  --policy.repo_id=dummy \
  --policy.n_action_steps=20 \
  --policy.chunk_size=20

