# test_aug_pipeline_save_imgs.py
import os
from pprint import pprint

import torch
from pathlib import Path

from PIL import Image

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.agibot_utils.AugmentedDataset import AugmentedDataset
import lerobot.agibot_utils.aug_funcs as aug_funcs
from lerobot.datasets.utils import load_language_data, load_episodes
# ---- config ----
REPO_ID = "/work/data/Manipulation-SimData_LeRobot_all"
EPISODES = [0, 10, 11, 23]  # small subset for quick test
OUTDIR = "aug_test_outputs"
os.makedirs(OUTDIR, exist_ok=True)
from torch.utils.data import DataLoader, SubsetRandomSampler


dataset = LeRobotDataset(REPO_ID, episodes=EPISODES)
indices = list(range(len(dataset)))
sampler = SubsetRandomSampler(indices)

orig_loader = DataLoader(dataset, batch_size=3, sampler=sampler, num_workers=0)
for batch in orig_loader:
    print("task (original):", batch['task'])
    print("english_action_text:", batch['english_action_text'])
    break