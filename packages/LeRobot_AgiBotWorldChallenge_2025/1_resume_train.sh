#!/bin/bash

python -m lerobot.scripts.train_distributed_gyb \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=/data/v-yuboguo/Manipulation-SimData_test_16dim/task/clear_table_in_the_restaurant_16dim \
  --output_dir=outputs/train/smolvla_clear_table_in_the_restaurant_from_base \
  --save_freq=10000 \
  --policy.repo_id=dummy
