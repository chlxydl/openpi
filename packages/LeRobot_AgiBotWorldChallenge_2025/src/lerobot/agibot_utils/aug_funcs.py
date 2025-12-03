# lerobot/agibot_utils/aug_funcs.py
import random
from typing import Tuple, List
import torch
import numpy as np
from PIL import Image
from typing import Optional
import torch

def _get_camera_keys(sample: dict, max_keys: int | None = None) -> List[str]:
    """
    返回样本中所有可能的 camera key（优先匹配常见的 observation.images.* 风格）。
    如果 dataset.meta 可用且 sample 中含 meta_info，则尝试使用 meta_info 内的 camera keys（调用者可自行扩展）。
    Args:
        sample: dict-like sample
        max_keys: 限制返回数量
    Returns:
        list[str]: 可能的 camera key 列表（至少会返回一个，若找不到则抛错）
    """
    # 首先尝试匹配常见的 camera key 前缀 (例如 'observation.images.' 或包含 'image'/'images' 的 key)
    candidate_keys = []
    for k in getattr(sample, "keys", lambda: [])():
        if k in ("episode_id", "episode_dir", "job_description", "used_cam_cfg", "meta_info"):
            continue
        if "image" in k or "images" in k or "camera" in k or k.endswith("_color") or k.endswith("_rgb"):
            candidate_keys.append(k)

    # 如果找到匹配的 camera-like keys，就返回它们
    if len(candidate_keys) > 0:
        keys = candidate_keys
    else:
        # 回退：返回 sample 中第一个非 meta key
        keys = [k for k in getattr(sample, "keys", lambda: [])() if k not in ("episode_id", "episode_dir", "job_description", "used_cam_cfg", "meta_info")]

    if len(keys) == 0:
        # 进一步尝试从 nested meta_info 中找 camera 配置
        if "meta_info" in sample and isinstance(sample["meta_info"], dict):
            cam_cfg = sample["meta_info"].get("cam_cfg")
            if isinstance(cam_cfg, dict):
                keys = list(cam_cfg.keys())

    if len(keys) == 0:
        raise RuntimeError("Could not determine camera keys in sample. Sample keys: " + ", ".join(list(getattr(sample, "keys", lambda: [])())))

    if max_keys is not None:
        return keys[:max_keys]
    return keys


def _get_primary_camera_key(sample: dict, prefer: str | None = None) -> str:
    """
    返回单个主要 camera key（第一个匹配到的）。
    prefer: 可传入偏好前缀（例如 'observation.images'）来尝试优先选择。
    """
    keys = _get_camera_keys(sample)
    if prefer:
        for k in keys:
            if k.startswith(prefer):
                return k
    return keys[0]


# -----------------------
# Image augmentations
# -----------------------
def random_flip(sample: dict, p: float = 0.5) -> dict:
    """随机水平翻转。对整个 image tensor 或 image list 做一致翻转（若存在多帧）。"""
    if p <= 0.0:
        return sample
    camera_key = _get_primary_camera_key(sample)
    if camera_key not in sample:
        raise KeyError(f"random_flip: camera_key '{camera_key}' not found in sample keys.")
    imgs = sample[camera_key]  # e.g. shape (T, C, H, W) or (C,H,W) or (B,T,C,H,W) etc.
    if torch.rand(1).item() < p:
        # only flip width dimension: works for (..., H, W) => flip last dim
        # ensure tensor type
        if isinstance(imgs, torch.Tensor):
            imgs = imgs.flip(-1)
        else:
            # try to convert numpy -> torch, flip, convert back
            try:
                imgs = torch.from_numpy(np.array(imgs)).flip(-1)
            except Exception:
                raise TypeError("random_flip: unsupported image type for flipping")
    sample[camera_key] = imgs
    return sample


def add_gaussian_noise(sample: dict, std: float = 0.01, clip: bool = True) -> dict:
    """给图像加高斯噪声（假定为 torch.Tensor 或能转为 torch.Tensor）。"""
    camera_key = _get_primary_camera_key(sample)
    if camera_key not in sample:
        raise KeyError(f"add_gaussian_noise: camera_key '{camera_key}' not found in sample keys.")
    imgs = sample[camera_key]
    if not isinstance(imgs, torch.Tensor):
        try:
            imgs = torch.from_numpy(np.array(imgs))
        except Exception:
            raise TypeError("add_gaussian_noise: unsupported image type")
    noise = torch.randn_like(imgs) * float(std)
    imgs = imgs + noise
    if clip:
        imgs = torch.clamp(imgs, 0.0, 1.0)
    sample[camera_key] = imgs
    return sample


def random_brightness(sample: dict,
                      factor_range: Tuple[float, float] = (0.8, 1.2),
                      p: float = 0.5) -> dict:
    """
    随机调整亮度，带概率触发
    sample: dict，包含图像的 sample
    factor_range: 亮度因子范围
    p: 触发增强的概率
    """
    # 概率触发
    if random.random() > p:
        return sample  # 不增强

    # 找主相机 key
    camera_key = _get_primary_camera_key(sample)
    if camera_key not in sample:
        raise KeyError(f"random_brightness: camera_key '{camera_key}' not found in sample keys.")

    imgs = sample[camera_key]

    # 转为 tensor
    if not isinstance(imgs, torch.Tensor):
        try:
            imgs = torch.from_numpy(np.array(imgs)).float()
        except Exception:
            raise TypeError("random_brightness: unsupported image type")

    # 随机亮度因子
    factor = random.uniform(*factor_range)
    imgs = imgs * factor
    imgs = torch.clamp(imgs, 0.0, 1.0)

    sample[camera_key] = imgs
    return sample


def image_dropout(obs, image_keys, drop_rate=0.5):
    """
    随机将输入 obs 中的一些图像置零（dropout），保持字典格式不变
    obs: dict, 包含多个图像 key
    image_keys: list[str], 需要dropout的图像key
    drop_rate: float, dropout 概率
    """
    pad_mask = torch.rand(len(image_keys)) > drop_rate  # True 表示保留，False 表示drop
    obs["pad_mask_dict"] = {}

    for i, key in enumerate(image_keys):
        img = obs[key]
        obs["pad_mask_dict"][key] = pad_mask[i]

        if not pad_mask[i]:  # 需要drop
            obs[key] = torch.zeros_like(img)

    return obs

# -----------------------
# State / action augmentations (示例)
# -----------------------
def jitter_state_gaussian(sample: dict, key: str = "observation.state", std: float = 0.0) -> dict:
    """对状态向量添加高斯噪声（如果存在）。"""
    if key in sample:
        s = sample[key]
        if isinstance(s, torch.Tensor):
            noise = torch.randn_like(s) * float(std)
            sample[key] = s + noise
        else:
            sample[key] = s + np.random.randn(*s.shape).astype(np.float32) * std
    return sample


def get_instruction(task_name):

    if task_name == "heat_the_food_in_the_microwave":
        lang = "Pick up the yellow functional beverage can on the table with the left arm.;Threw the yellow functional beverage can into the trash can with the left arm.;Pick up the green carbonated beverage can on the table with the right arm.;Threw the green carbonated beverage can into the trash can with the right arm."
    elif task_name == "restock_supermarket_items":
        lang = "Pick up the brown plum juice from the restock box with the right arm.;Place the brown plum juice on the shelf where the brown plum juice is located with the right arm."
    elif task_name == "clear_table_in_the_restaurant":
        lang = "Pick up the bowl on the table near the right arm with the right arm.;Place the bowl on the plate on the table with the right arm."
    elif task_name == "stamp_the_seal":
        lang = "Pick up the stamp from the ink pad on the table with the right arm.;Stamp the document on the table with the stamp in the right arm.;Place the stamp into the ink pad on the table with the right arm."
    elif task_name == "pack_in_the_supermarket":
        lang = "Pick up the grape juice on the table with the right arm.;Put the grape juice into the felt bag on the table with the right arm."
    elif task_name == "heat_the_food_in_the_microwave_v2":  # 给新版单独标记
        lang = "Open the door of the microwave oven with the right arm.;Pick up the plate with bread on the table with the right arm.;Put the plate containing bread into the microwave oven with the right arm.;Push the plate that was not placed properly into the microwave oven the right arm.;Close the door of the microwave oven with the left arm.;Press the start button on the right side of the microwave oven with the right arm."
    elif task_name == "open_drawer_and_store_items":
        lang = "Pull the top drawer of the drawer cabinet with the right arm.;Pick up the Rubik's Cube on the drawer cabinet with the right arm.;Place the Rubik's Cube into the drawer with the right arm.;Push the top drawer of the drawer cabinet with the right arm."
    elif task_name == "pack_moving_objects_from_conveyor":
        lang = "Pick up the hand cream from the conveyor belt with the right arm;Place the hand cream held in the right arm into the box on the table"
    elif task_name == "pickup_items_from_the_freezer":
        lang = "Open the freezer door with the right arm;Pick up the caviar from the freezer with the right arm;Place the caviar held in the right arm into the shopping cart;Close the freezer door with both arms"
    elif task_name == "make_a_sandwich":
        lang = "Pick up the bread slice from the toaster on the table with the right arm;Place the picked bread slice into the plate on the table with the right arm;Pick up the ham slice from the box on the table with the left arm;Place the picked ham slice onto the bread slice in the plate on the table with the left arm;Pick up the lettuce slice from the box on the table with the right arm;Place the picked lettuce slice onto the ham slice in the plate on the table with the right arm;Pick up the bread slice from the toaster on the table with the right arm;Place the bread slice onto the lettuce slice in the plate on the table with the right arm"
    else:
        raise ValueError(f"task '{task_name}' does not exist")

    return lang

# -----------------------
# Language / prompt augmentation
# -----------------------
def runtime_prompt_generation(sample: dict, prompt_mode_list=None) -> dict:
    """
    以函数形式为 sample 生成 prompt 并写回 sample["task"]。
    兼容 aug_cfg 中把函数作为 type 且通过 params 传入 prompt_mode_list 的用法：
      {"type": runtime_prompt_generation, "params": {"prompt_mode_list":[0,1,2,3]}}
    或者直接把工厂的返回值放进 aug_cfg：
      {"type": make_runtime_prompt_generator([0,1,2,3]), "params": {}}
    """
    if prompt_mode_list is None:
        prompt_mode_list = [0]
    # 保证是 list
    prompt_mode_list = list(prompt_mode_list)

    # 随机选择 prompt 模式
    prompt_mode = random.choice(prompt_mode_list)

    # 安全读取字段
    task_val = sample.get("task", "")
    english_action_text = sample.get("english_action_text", "")
    # print("english_action_text in runtime_prompt_generation:", english_action_text)
    # english_action_text 在原数据里可能是 list/array（batch）或字符串，尽量取简单字符串
    if isinstance(english_action_text, (list, tuple)) and len(english_action_text) > 0:
        eng_text = english_action_text[0]
    else:
        eng_text = english_action_text

    # 处理 task 类似情况（有时为 list）
    if isinstance(task_val, (list, tuple)) and len(task_val) > 0:
        task_str = task_val[0]
    else:
        task_str = task_val

    # 构建 final_prompt
    if prompt_mode == 0:
        prompt = f"What action should the robot take to {task_str}?"
    elif prompt_mode == 1:
        prompt = f"The robot is performing the step of {eng_text}."
    elif prompt_mode == 2:
        prompt = f"What action should the robot take to {task_str}? The robot is performing the step of {eng_text}."
    elif prompt_mode == 3:
        # 调用 get_instruction 来获得任务的完整语言描述
        try:
            task_instruction = get_instruction(task_str)
        except Exception:
            task_instruction = task_str
        prompt = f"The robot is performing the step of {task_instruction}"
    else:
        raise IndexError(f"invalid prompt_mode: {prompt_mode}")

    # 将生成的 prompt 写回 sampl
    sample["task"] = prompt
    return sample
