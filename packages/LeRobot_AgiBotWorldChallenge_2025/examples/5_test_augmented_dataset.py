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
REPO_ID = "/home/v-yuboguo/LeRobot_AgiBotWorldChallenge_2025/datasets"
EPISODES = [0, 10, 11, 23]  # small subset for quick test
OUTDIR = "aug_test_outputs"
os.makedirs(OUTDIR, exist_ok=True)

from torchvision.transforms.functional import to_pil_image
import numpy as np
def save_one_image(batch, outdir, prefix):
    """
    更稳健的保存函数：
      - 自动发现相机键（优先使用 aug_funcs._get_camera_keys）
      - 处理常见张量形状：(B,C,H,W), (C,H,W), (T,C,H,W), (B,T,C,H,W)
      - batch 为 DataLoader 返回（含 batch dim），会取第0个样例 / 第0帧
    """
    os.makedirs(outdir, exist_ok=True)

    # 先尝试用 aug_funcs 的方法发现 keys
    try:
        cam_keys = aug_funcs._get_camera_keys(batch)
    except Exception:
        # 如果 batch 不是单个 sample 而是包含 batch-dim（DataLoader返回），
        # _get_camera_keys 可能找不到 keys。尝试用 batch.keys() 发现相机-like keys。
        cam_keys = [k for k in batch.keys() if ("image" in k or "images" in k or "camera" in k)]

    if len(cam_keys) == 0:
        print("[WARN] no camera keys found in batch.")
        return

    for key in cam_keys:
        if key not in batch:
            continue
        img_val = batch[key]

        # 如果是 DataLoader 输出的 batch（通常第0个为样例）
        # 支持形状：(B,C,H,W), (B,T,C,H,W), (C,H,W), (T,C,H,W), (B,T,C,H,W)
        try:
            # if tensor
            if isinstance(img_val, torch.Tensor):
                img = img_val.detach().cpu()
                # (B, ...) -> take first batch element
                if img.ndim >= 4 and img.shape[0] > 1:
                    # could be (B,C,H,W) or (B,T,C,H,W)
                    img = img[0]
                # if now (T,C,H,W) -> take first frame
                if img.ndim == 4:
                    img = img[0]
                # now expected (C,H,W)
                if img.ndim == 3:
                    # ensure float->uint8 conversion
                    arr = img.permute(1,2,0).numpy()
                    if arr.dtype == np.float32 or arr.dtype == np.float64:
                        arr = np.clip(arr, 0.0, 1.0)
                        arr = (arr * 255.0).astype(np.uint8)
                    else:
                        arr = arr.astype(np.uint8)
                    pil = Image.fromarray(arr)
                elif img.ndim == 2:
                    arr = img.numpy()
                    if arr.dtype == np.float32 or arr.dtype == np.float64:
                        arr = np.clip(arr, 0.0, 1.0)
                        arr = (arr * 255.0).astype(np.uint8)
                    pil = Image.fromarray(arr)
                else:
                    raise ValueError(f"Unsupported tensor ndim={img.ndim} for key {key}")
            else:
                # non-tensor: try numpy or PIL
                if isinstance(img_val, np.ndarray):
                    arr = img_val
                    if arr.ndim == 3 and arr.shape[0] in (1,3):  # maybe CHW
                        if arr.shape[0] in (1,3):
                            if arr.shape[0] == 3:
                                arr = arr.transpose(1,2,0)
                            else:
                                arr = arr.squeeze(0)
                    if arr.dtype == np.float32 or arr.dtype == np.float64:
                        arr = np.clip(arr, 0.0, 1.0)
                        arr = (arr * 255.0).astype(np.uint8)
                    pil = Image.fromarray(arr)
                elif isinstance(img_val, Image.Image):
                    pil = img_val
                else:
                    print(f"[WARN] Unsupported image type {type(img_val)} for key {key}, skipping.")
                    continue

            safe_key = key.replace("/", "_").replace(".", "_")
            out_path = os.path.join(outdir, f"{prefix}_{safe_key}.png")
            pil.save(out_path)
        except Exception as e:
            print(f"[WARN] failed to save key {key}: {e}")
            continue

def save_side_by_side(orig_sample, aug_sample, outdir, prefix):
    """
    将同一样本的原始图和增强图左右拼接保存
    """
    os.makedirs(outdir, exist_ok=True)

    # 使用原来的 save_one_image 的逻辑来获取 PIL 图像
    def tensor_or_array_to_pil(img_val):
        if isinstance(img_val, torch.Tensor):
            img = img_val.detach().cpu()
            if img.ndim == 4:  # (T,C,H,W) or (B,C,H,W)
                img = img[0]
            if img.ndim == 3:
                arr = img.permute(1,2,0).numpy()
            elif img.ndim == 2:
                arr = img.numpy()
            else:
                raise ValueError(f"Unsupported tensor shape {img.shape}")
        elif isinstance(img_val, np.ndarray):
            arr = img_val
            if arr.ndim == 3 and arr.shape[0] in (1,3):  # CHW -> HWC
                arr = arr.transpose(1,2,0) if arr.shape[0]==3 else arr.squeeze(0)
        elif isinstance(img_val, Image.Image):
            return img_val
        else:
            raise TypeError(f"Unsupported image type {type(img_val)}")
        
        if arr.dtype in (np.float32, np.float64):
            arr = np.clip(arr, 0.0, 1.0)
            arr = (arr*255).astype(np.uint8)
        return Image.fromarray(arr)

    # 默认使用第一个相机键
    cam_keys = [k for k in orig_sample.keys() if "image" in k or "images" in k]
    if len(cam_keys) == 0:
        print("[WARN] no camera keys found.")
        return
    key = cam_keys[0]

    img_orig = tensor_or_array_to_pil(orig_sample[key])
    img_aug  = tensor_or_array_to_pil(aug_sample[key])

    # 拼接左右
    w, h = img_orig.size
    new_img = Image.new("RGB", (w*2, h))
    new_img.paste(img_orig, (0,0))
    new_img.paste(img_aug, (w,0))

    out_path = os.path.join(outdir, f"{prefix}_side_by_side.png")
    new_img.save(out_path)
    print(f"Saved side-by-side image to {out_path}")

# 打印 batch 的 key 和 shape/长度
def print_batch_info(batch, name="Batch"):
    print(f"--- {name} ---")
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"{k}: Tensor, shape={v.shape}, dtype={v.dtype}")
        elif isinstance(v, list):
            print(f"{k}: list, length={len(v)}")
            if len(v) > 0:
                print(f"  first element type: {type(v[0])}")
        else:
            print(f"{k}: type={type(v)}, value={v}")
    print("---------------------\n")
# REPO_ID_PATH = Path("/home/v-yuboguo/LeRobot_AgiBotWorldChallenge_2025/datasets")

# episode_tasks, episode_to_index = load_language_data(REPO_ID_PATH)
# # 查看第0号episode的内容
# print(episode_tasks[0]["tasks"])           # 任务名称列表
# print(episode_tasks[0]["action_config"])   # 动作配置列表
# episode_index = episode_to_index[episode_tasks[0]["tasks"][0]]
# print("对应的 episode_index:", episode_index)

# episodes = load_episodes(REPO_ID_PATH)
# # 如果列表太长，可以逐行打印
# episodes = load_episodes(REPO_ID_PATH)

# # 打印每个 episode 的完整字典
# for ep_index, ep_data in episodes.items():
#     print(f"Episode {ep_index}:")
#     print(ep_data)
#     print("--------")

# ---- build base dataset ----
dataset = LeRobotDataset(REPO_ID, episodes=EPISODES)
# print("Loaded base dataset")
# if hasattr(dataset, "meta"):
#     print("Dataset meta camera_keys:", getattr(dataset.meta, "camera_keys", None))

# ---- augmentation pipeline (parameterized) ----
aug_cfg = [
    {"type": aug_funcs.random_brightness, "params": {"factor_range": (0.5, 2.0), "p": 1.0}},
    # {"type": aug_funcs.add_gaussian_noise, "params": {"std": 0.02, "clip": True}},
    {"type": aug_funcs.image_dropout, "params": {
        "image_keys": ["observation.images.head", "observation.images.hand_left", "observation.images.hand_right"],
        "drop_rate": 0.1
    }},
    {"type": aug_funcs.runtime_prompt_generation, "params": {"prompt_mode_list": [0, 1, 2, 3]}},
]

from torch.utils.data import DataLoader, SubsetRandomSampler
import random

# 生成一组固定随机索引,保证两个loader获取的相同
indices = list(range(len(dataset)))
random_seed = 42
rng = random.Random(random_seed)
rng.shuffle(indices)

sampler = SubsetRandomSampler(indices)

# ---- wrap dataset with AugmentedDataset ----
aug_dataset = AugmentedDataset(dataset, augmentations=aug_cfg, enable_augment=True)
# print("Wrapped dataset with AugmentedDataset (enable_augment=True)")

# ---- dataloaders ----
orig_loader = DataLoader(dataset, batch_size=2, sampler=sampler, num_workers=0)
aug_loader = DataLoader(aug_dataset, batch_size=2, sampler=sampler, num_workers=0)

# 原始 loader
for batch in orig_loader:
    print("task (original):", batch['task'])
    print_batch_info(batch, "Original batch")
    break

# 增强 loader
for batch in aug_loader:
    print("task (augmented):", batch['task'])
    print_batch_info(batch, "Augmented batch")
    break
# idx = 0
# orig_sample = dataset[idx]
# aug_sample  = aug_dataset[idx]
# save_side_by_side(orig_sample, aug_sample, OUTDIR, f"sample{idx}")


