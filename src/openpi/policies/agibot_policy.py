import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model
import copy
import torch

import json
import dataclasses
import numpy as np
def make_libero_example() -> dict:
    """Creates a random input example for the Libero policy."""
    return {
        "observation/state": np.random.rand(8),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "do something",
    }



def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image



def _extract_task_name(task_field: object) -> str:
    """
    从 data['task'] 提取干净的 task 名。
    支持输入形式：
      - "pack_in_the_supermarket | "（字符串）
      - '{"task": "pack_in_the_supermarket | "}'（json 字符串）
      - {"task": "pack_in_the_supermarket | "}（dict）
    返回第一个 '|' 左侧、去掉首尾空白的 token，例如 "pack_in_the_supermarket"。
    """
    # 规范化为字符串或 dict
    if isinstance(task_field, dict):
        raw = task_field.get("task", "")
    elif isinstance(task_field, str):
        raw = task_field.strip()
        # 尝试解析 JSON 字符串（防止有人把 dict 当字符串传）
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict) and "task" in parsed:
                raw = parsed["task"]
        except Exception:
            # 不是 JSON 就继续使用原始字符串
            pass
    else:
        raw = str(task_field)

    # 拆分 '|' 并取第一个片段，去除左右空白
    cleaned = raw.split("|", 1)[0].strip()
    return cleaned

import random

# def _runtime_prompt_generation(data: dict, prompt_mode_list=None) -> dict:
#     """
#     统一生成 prompt，兼容新的 data 格式:
#       - data["prompt"]: 原始 prompt 名 (e.g., "pack_in_the_supermarket")
#       - data["english_action_text"]: 机器人动作描述
#     并写回 data["task"] 作为最终的自然语言 prompt。
#     """
#     if prompt_mode_list is None:
#         prompt_mode_list = [0]
#     prompt_mode_list = list(prompt_mode_list)

#     # 随机选择 prompt 模式
#     prompt_mode = random.choice(prompt_mode_list)
#     # print("[Before] data['prompt']:", data.get("prompt", ""))
#     # print("[Before] data['english_action_text']:", data.get("english_action_text", ""))

#     # 先提取 task 名
#     task_name = ""
#     if "prompt" in data:
#         task_raw = data["prompt"]
#         task_name = _extract_task_name(task_raw)
#         try:
#             task_name = get_instruction(task_name)  # 转自然语言
#         except Exception:
#             pass

#     # 再提取 english_action_text
#     english_action_text = data.get("english_action_text", "")
#     if isinstance(english_action_text, (list, tuple)) and len(english_action_text) > 0:
#         eng_text = english_action_text[0]
#     else:
#         eng_text = english_action_text

#     # 根据模式生成 prompt
#     if prompt_mode == 0:
#         prompt = f"What action should the robot take to {task_name}?"
#     elif prompt_mode == 1:
#         prompt = f"The robot is performing the step of {eng_text}."
#     elif prompt_mode == 2:
#         prompt = f"What action should the robot take to {task_name}? The robot is performing the step of {eng_text}."
#     elif prompt_mode == 3:
#         try:
#             task_instruction = get_instruction(task_name)
#         except Exception:
#             task_instruction = task_name
#         prompt = f"The robot is performing the step of {task_instruction}"
#     else:
#         raise IndexError(f"invalid prompt_mode: {prompt_mode}")

#     # 写回 data["task"]，保留 english_action_text
#     data["prompt"] = prompt
#     if eng_text:
#         data["english_action_text"] = eng_text

#     # ---------- 转换后打印 ----------
#     # print("[After] data['prompt']:", data["prompt"])
#     # print("[After] data['english_action_text']:", data["english_action_text"])

#     return data

def _runtime_prompt_generation(data: dict, prompt_mode_list=None) -> dict:
    """
    统一生成 prompt，兼容新的 data 格式:
      - data["prompt"]: 原始 prompt 名 (e.g., "pack_in_the_supermarket")
      - data["english_action_text"]: 机器人动作描述
    并写回 data["task"] 作为最终的自然语言 prompt。
    """
    if prompt_mode_list is None:
        prompt_mode_list = [0]
    prompt_mode_list = list(prompt_mode_list)

    # 随机选择 prompt 模式
    prompt_mode = random.choice(prompt_mode_list)
    # print("[Before] data['prompt']:", data.get("prompt", ""))
    # print("[Before] data['english_action_text']:", data.get("english_action_text", ""))

    # 先提取 task 名
    task_name = ""
    if "prompt" in data:
        task_raw = data["prompt"]
        task_name = _extract_task_name(task_raw)
        try:
            task_name = get_instruction(task_name)  # 转自然语言
        except Exception:
            pass

    # 再提取 english_action_text
    english_action_text = data.get("english_action_text", "")
    if isinstance(english_action_text, (list, tuple)) and len(english_action_text) > 0:
        eng_text = english_action_text[0]
    else:
        eng_text = english_action_text

    # 根据模式生成 prompt
    if prompt_mode == 0:
        prompt = f"What action should the robot take to {task_name}?"
    elif prompt_mode == 1:
        prompt = f"The robot is performing the step of {eng_text}."
    elif prompt_mode == 2:
        prompt = f"What action should the robot take to {task_name}? The robot is performing the step of {eng_text}."
    elif prompt_mode == 3:
        try:
            task_instruction = get_instruction(task_name)
        except Exception:
            task_instruction = task_name
        prompt = f"The robot is performing the step of {task_instruction}"
    else:
        raise IndexError(f"invalid prompt_mode: {prompt_mode}")

    # 写回 data["task"]，保留 english_action_text
    data["prompt"] = prompt
    if eng_text:
        data["english_action_text"] = eng_text

    # ---------- 转换后打印 ----------
    # print("[After] data['prompt']:", data["prompt"])
    # print("[After] data['english_action_text']:", data["english_action_text"])

    return data
@dataclasses.dataclass(frozen=True)
class AgibotInputs(transforms.DataTransformFn):
    """
    This class is used to convert inputs to the model to the expected format. It is used for both training and inference.

    For your own dataset, you can copy this class and modify the keys based on the comments below to pipe
    the correct elements of your dataset into the model.
    """

    # The action dimension of the model. Will be used to pad state and actions for pi0 model (not pi0-FAST).
    # Do not change this for your own dataset.
    action_dim: int

    # Determines which model will be used.
    # Do not change this for your own dataset.
    model_type: _model.ModelType = _model.ModelType.PI0
    is_compute_norm: bool = False
    input_mode: str = 'joint'  # 'ee', 'joint', 'ee_joint'. Determines which inputs to use from the dataset.

    def __call__(self, data: dict) -> dict:
        assert self.input_mode == 'joint'
        # We only mask padding for pi0 model, not pi0-FAST. Do not change this for your own dataset.
        mask_padding = self.model_type == _model.ModelType.PI0
        # print("data.keys()\n", data.keys())
        # We pad the proprioceptive input to the action dimension of the model.
        # For pi0-FAST, we don't pad the state. For Libero, we don't need to differentiate
        # since the pi0-FAST action_dim = 7, which is < state_dim = 8, so pad is skipped.
        # Keep this for your own dataset, but if your dataset stores the proprioceptive input
        # in a different key than "observation/state", you should change it below.
        state = data['state'].clone()
        # gripper = state[:2]
        # ee_quat = state[2:10]
        # ee_pos = state[10:16]
        # joint = state[16:30]
        joint = state[:8]

        # from scipy.spatial.transform import Rotation as R
        # r = R.from_quat(ee_quat.numpy().reshape(2,-1))
        # state_ee_euler = r.as_euler('xyz', degrees=False).reshape(-1)
        if self.input_mode == 'joint':
            data["state"][:8] = joint
            data["state"][8:] = 0.0
        elif self.input_mode == 'ee':
            data['state'][2:8] = ee_pos
            data['state'][8:16] = ee_quat
            data['state'][16:] = 0.0
        elif self.input_mode == 'ee_joint':
            data['state'][2:16] = joint
            data['state'][16:22] = ee_pos
            data['state'][22:] = ee_quat
        state = transforms.pad_to_dim(data["state"], self.action_dim)

        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference.
        # Keep this for your own dataset, but if your dataset stores the images
        # in a different key than "observation/image" or "observation/wrist_image",
        # you should change it below.
        # Pi0 models support three image inputs at the moment: one third-person view,
        # and two wrist views (left and right). If your dataset does not have a particular type
        # of image, e.g. wrist images, you can comment it out here and replace it with zeros like we do for the
        # right wrist image below.
        if not self.is_compute_norm:
            head_image = _parse_image(data["observation.images.head"])
            hand_left_image = _parse_image(data["observation.images.hand_left"])
            hand_right_image = _parse_image(data["observation.images.hand_right"])

            # Create inputs dict. Do not change the keys in the dict below.
            inputs = {
                "state": state,
                "image": {
                    "base_0_rgb": head_image,
                    "left_wrist_0_rgb": hand_left_image,
                    # Pad any non-existent images with zero-arrays of the appropriate shape.
                    "right_wrist_0_rgb": hand_right_image,
                },
                "image_mask": {
                    "base_0_rgb": np.True_,
                    "left_wrist_0_rgb": np.True_,
                    # Mask any non-existent images with False (if ``mask_padding`` is True).
                    "right_wrist_0_rgb": np.True_,
                },
                "prompt": data["prompt"],
            }
        else:
            inputs = {
                "state": state
            }


        # Pad actions to the model action dimension. Keep this for your own dataset.
        # Actions are only available during training.
        if "actions" in data:
            # We are padding to the model action dim.
            # For pi0-FAST, this is a no-op (since action_dim = 7).
            actions = data['actions'].clone()
            # gripper = actions[:, :2]
            # ee_quat = actions[:, 2:10]
            # ee_pos = actions[:, 10:16]
            # joint = actions[:, 16:30]
            joint = actions[:, :8]

            # from scipy.spatial.transform import Rotation as R
            # r = R.from_quat(ee_quat.numpy().reshape(-1,4))
            # ee_euler = r.as_euler('xyz', degrees=False).reshape(-1, 6)
            if self.input_mode == 'joint':
                data["actions"][:, :8] = joint
                data["actions"][:, 8:] = 0.0
            elif self.input_mode == 'ee':
                ee_euler_diff = ee_euler - state_ee_euler.reshape(1,-1)
                ee_euler_diff = (ee_euler_diff + np.pi) % (2 * np.pi) - np.pi
                data['actions'][:, 2:8] = ee_pos
                data['actions'][:, 8:14] = torch.Tensor(ee_euler_diff)
                data['actions'][:, 14:] = 0.0
            elif self.input_mode == 'ee_joint':
                ee_euler_diff = ee_euler - state_ee_euler.reshape(1,-1)
                ee_euler_diff = (ee_euler_diff + np.pi) % (2 * np.pi) - np.pi  # wrap to [-pi, pi]
                data['actions'][:, 2:16] = joint
                data['actions'][:, 16:22] = ee_pos
                data['actions'][:, 22:28] = torch.Tensor(ee_euler_diff)
                data['actions'][:, 28:] = 0.0

            # joint_action = data["actions"][:,16:30]
            # data["actions"][:,2:16] = joint_action
            # data["actions"][:,16:] = 0.0
            actions = transforms.pad_to_dim(data["actions"], self.action_dim)
            inputs["actions"] = actions

        # Pass the prompt (aka language instruction) to the model.
        # Keep this for your own dataset (but modify the key if the instruction is not
        # stored in "prompt"; the output dict always needs to have the key "prompt").

        # --- 处理 prompt 字段：清理并映射到 get_instruction ---
        
        # _runtime_prompt_generation(data, prompt_mode_list=[0,1,2,3])
        
        # inputs["prompt"] = data["prompt"]
        # if "prompt" in data:
        #     prompt_raw = data["prompt"]
        #     prompt_name = _extract_task_name(prompt_raw)  # 得到如 "pack_in_the_supermarket"
        #     try:
        #         # 尝试用映射函数获得自然语言指令
        #         inputs["prompt"] = get_instruction(prompt_name)
        #     except Exception:
        #         # 如果映射失败（例如 prompt_name 不存在于映射表），回退到干净的 prompt 名本身
        #         inputs["prompt"] = prompt_name

        # print("inputs['prompt']:", inputs["prompt"])
        # if "english_action_text" in data:
        #     print("data['english_action_text']:", data["english_action_text"])
        #     inputs["english_action_text"] = data["english_action_text"]
        # print("data['prompt']:", data["prompt"])
        # print("inputs['prompt']:", inputs["prompt"])
        return inputs


@dataclasses.dataclass(frozen=True)
class AgibotOutputs(transforms.DataTransformFn):
    """
    This class is used to convert outputs from the model back the the dataset specific format. It is
    used for inference only.

    For your own dataset, you can copy this class and modify the action dimension based on the comments below.
    """
    output_mode: str = 'joint'  # 'ee', 'joint', 'ee_joint'. Determines which outputs to return from the model.

    def __call__(self, data: dict) -> dict:
        # Only return the first N actions -- since we padded actions above to fit the model action
        # dimension, we need to now parse out the correct number of actions in the return dict.
        # For Libero, we only return the first 7 actions (since the rest is padding).
        # For your own dataset, replace `7` with the action dimension of your dataset.
        if self.output_mode == 'joint':
            return {"actions": np.asarray(data["actions"][:, :8])}
        elif self.output_mode == 'ee':
            return {"actions": np.asarray(data["actions"][:, :14])}
        elif self.output_mode == 'ee_joint':
            return {"actions": np.asarray(data["actions"][:, :28])}

def get_instruction(task_name):
    if task_name == "iros_clear_the_countertop_waste":
        lang = "Pick up the yellow functional beverage can on the table with the left arm. Threw the yellow functional beverage can into the trash can with the left arm. Pick up the green carbonated beverage can on the table with the right arm. Threw the green carbonated beverage can into the trash can with the right arm."
    elif task_name == "iros_restock_supermarket_items":
        lang = "Pick up the brown plum juice from the restock box with the right arm. Place the brown plum juice on the shelf where the brown plum juice is located with the right arm."
    elif task_name == "iros_clear_table_in_the_restaurant":
        lang = "Pick up the bowl on the table near the right arm with the right arm. Place the bowl on the plate on the table with the right arm."
    elif task_name == "iros_stamp_the_seal":
        lang = "Pick up the stamp from the ink pad on the table with the right arm. Stamp the document on the table with the stamp in the right arm. Place the stamp into the ink pad on the table with the right arm."
    elif task_name == "iros_pack_in_the_supermarket":
        lang = "Pick up the grape juice on the table with the right arm. Put the grape juice into the felt bag on the table with the right arm."
    elif task_name == "iros_heat_the_food_in_the_microwave":
        lang = "Open the door of the microwave oven with the right arm. Pick up the plate with bread on the table with the right arm. Put the plate containing bread into the microwave oven with the right arm. Push the plate that was not placed properly into the microwave oven the right arm. Close the door of the microwave oven with the left arm. Press the start button on the right side of the microwave oven with the right arm."
    elif task_name == "iros_open_drawer_and_store_items":
        lang = "Pull the top drawer of the drawer cabinet with the right arm. Pick up the Rubik's Cube on the drawer cabinet with the right arm. Place the Rubik's Cube into the drawer with the right arm. Push the top drawer of the drawer cabinet with the right arm."
    elif task_name == "iros_pack_moving_objects_from_conveyor":
        lang = "Pick up the hand cream from the conveyor belt with the right arm. Place the hand cream held in the right arm into the box on the table"
    elif task_name == "iros_pickup_items_from_the_freezer":
        lang = "Open the freezer door with the right arm. Pick up the caviar from the freezer with the right arm. Place the caviar held in the right arm into the shopping cart. Close the freezer door with both arms"
    elif task_name == "iros_make_a_sandwich":
        lang = "Pick up the bread slice from the toaster on the table with the right arm. Place the picked bread slice into the plate on the table with the right arm. Pick up the ham slice from the box on the table with the left arm. Place the picked ham slice onto the bread slice in the plate on the table with the left arm. Pick up the lettuce slice from the box on the table with the right arm. Place the picked lettuce slice onto the ham slice in the plate on the table with the right arm. Pick up the bread slice from the toaster on the table with the right arm. Place the bread slice onto the lettuce slice in the plate on the table with the right arm"
    else:
        raise ValueError("task does not exist")

    return lang