############################
# README： 这是推理一个动作版本
############################
import os
import sys
from pathlib import Path
# 保证可以找到 genie_sim_ros 和 lerobot
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent / 'src'))

import threading

import cv2
import numpy as np
import torch
import rclpy
from cv_bridge import CvBridge

import draccus
from dataclasses import dataclass

# from lerobot.policies.act.modeling_act import ACTPolicy
# from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.pi0.modeling_pi0 import PI0Policy

import time
import base64
import argparse

from genie_sim_ros_lerobot import SimROSNode

# def resize_img(img, width, height):
#     resized_img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
#     resized_img = np.array(resized_img)
#     return resized_img

#改输入为模型需要格式
# def resize_img(img, target_width=640, target_height=480, device='cuda'):
#     """
#     Resize image to fit in (target_width x target_height) while keeping aspect ratio,
#     and pad with black borders if needed.
#     """
#     h, w = img.shape[:2]

#     # 计算缩放比例
#     scale = min(target_width / w, target_height / h)
#     new_w = int(w * scale)
#     new_h = int(h * scale)

#     # 缩放
#     resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

#     # 创建黑色画布
#     canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)

#     # 把缩放后的图像放到中间
#     top = (target_height - new_h) // 2
#     left = (target_width - new_w) // 2
#     canvas[top:top + new_h, left:left + new_w] = resized_img
#     chw_img = canvas.transpose(2, 0, 1)
#     chw_img = torch.from_numpy(chw_img / 255.0).float().unsqueeze(0).to(device)
#     return chw_img

def resize_img(img: np.ndarray, target_width=640, target_height=480) -> np.ndarray:
    """
    Resize image to (target_width x target_height) while keeping aspect ratio,
    and pad with black borders if needed.
    返回仍是 HWC 格式的 uint8 图像。
    """
    h, w = img.shape[:2]

    # 计算缩放比例
    scale = min(target_width / w, target_height / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    # 缩放
    resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 创建黑色画布
    canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)

    # 把缩放后的图像放到中间
    top = (target_height - new_h) // 2
    left = (target_width - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = resized_img

    return canvas  # 仍是 uint8，HWC

def encode_image_to_base64(img: np.ndarray) -> str:
    """
    将图像编码为 base64 字符串。
    """
    _, buffer = cv2.imencode('.jpg', img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return img_base64

def decode_image_from_base64(img_base64: str) -> np.ndarray:
    """
    从 base64 字符串解码图像为 numpy array。
    """
    img_decoded = base64.b64decode(img_base64)
    img_array = np.frombuffer(img_decoded, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return img

def to_model_input(img: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    将 HWC 格式图像转换为模型输入格式：
    - HWC → CHW
    - [0, 255] → [0, 1] float32
    - 加 batch 维度
    - 转为 torch.Tensor 并移动到 device
    """
    chw_img = img.transpose(2, 0, 1)
    chw_img = torch.from_numpy(chw_img / 255.0).float().unsqueeze(0).to(device)
    return chw_img

def preprocess_image(img: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    主入口：从原始图像 → resize → base64 编码解码 → 模型输入张量
    """
    img = resize_img(img)
    img = decode_image_from_base64(encode_image_to_base64(img))
    return to_model_input(img, device)

def get_instruction(task_name):

    if task_name == "iros_clear_the_countertop_waste":
        lang = "Pick up the yellow functional beverage can on the table with the left arm.;Threw the yellow functional beverage can into the trash can with the left arm.;Pick up the green carbonated beverage can on the table with the right arm.;Threw the green carbonated beverage can into the trash can with the right arm."
    elif task_name == "iros_restock_supermarket_items":
        lang = "Pick up the brown plum juice from the restock box with the right arm.;Place the brown plum juice on the shelf where the brown plum juice is located with the right arm."
    elif task_name == "iros_clear_table_in_the_restaurant":
        lang = "Pick up the bowl on the table near the right arm with the right arm.;Place the bowl on the plate on the table with the right arm."
    elif task_name == "iros_stamp_the_seal":
        lang = "Pick up the stamp from the ink pad on the table with the right arm.;Stamp the document on the table with the stamp in the right arm.;Place the stamp into the ink pad on the table with the right arm."
    elif task_name == "iros_pack_in_the_supermarket":
        lang = "Pick up the grape juice on the table with the right arm.;Put the grape juice into the felt bag on the table with the right arm."
    elif task_name == "iros_heat_the_food_in_the_microwave":
        lang = "Open the door of the microwave oven with the right arm.;Pick up the plate with bread on the table with the right arm.;Put the plate containing bread into the microwave oven with the right arm.;Push the plate that was not placed properly into the microwave oven the right arm.;Close the door of the microwave oven with the left arm.;Press the start button on the right side of the microwave oven with the right arm."
    elif task_name == "iros_open_drawer_and_store_items":
        lang = "Pull the top drawer of the drawer cabinet with the right arm.;Pick up the Rubik's Cube on the drawer cabinet with the right arm.;Place the Rubik's Cube into the drawer with the right arm.;Push the top drawer of the drawer cabinet with the right arm."
    elif task_name == "iros_pack_moving_objects_from_conveyor":
        lang = "Pick up the hand cream from the conveyor belt with the right arm;Place the hand cream held in the right arm into the box on the table"
    elif task_name == "iros_pickup_items_from_the_freezer":
        lang = "Open the freezer door with the right arm;Pick up the caviar from the freezer with the right arm;Place the caviar held in the right arm into the shopping cart;Close the freezer door with both arms"
    elif task_name == "iros_make_a_sandwich":
        lang = "Pick up the bread slice from the toaster on the table with the right arm;Place the picked bread slice into the plate on the table with the right arm;Pick up the ham slice from the box on the table with the left arm;Place the picked ham slice onto the bread slice in the plate on the table with the left arm;Pick up the lettuce slice from the box on the table with the right arm;Place the picked lettuce slice onto the ham slice in the plate on the table with the right arm;Pick up the bread slice from the toaster on the table with the right arm;Place the bread slice onto the lettuce slice in the plate on the table with the right arm"
    else:
        raise ValueError("task does not exist")

    return lang

def get_sim_time(sim_ros_node):
    sim_time = sim_ros_node.get_clock().now().nanoseconds * 1e-9
    return sim_time

def make_policy():



    path_ckpt = "/root/workspace/main/AgiBot-World/checkpoints/pi0/pretrained_model" 

    print("path_ckpt:",path_ckpt)
    policy = PI0Policy.from_pretrained(pretrained_name_or_path = path_ckpt, local_files_only=True)

    print(policy.config.input_features)

    print("policy reset")
    policy.reset()

    return policy

def infer(cfg):
    # print(cfg)


    rclpy.init()
    sim_ros_node = SimROSNode()
    spin_thread = threading.Thread(target=rclpy.spin, args=(sim_ros_node,))
    spin_thread.start()

    bridge = CvBridge()
    count = 0
    SIM_INIT_TIME = 5
    last_ts_h = -1.0


    fps_start_time = None
    policy = make_policy()
    print("start loop")
    task_instruction = get_instruction(cfg.task_name)
    print("now task:",task_instruction)
    while rclpy.ok():
        img_h_raw = sim_ros_node.get_img_head()
        img_l_raw = sim_ros_node.get_img_left_wrist()
        img_r_raw = sim_ros_node.get_img_right_wrist()
        act_raw = sim_ros_node.get_joint_state()
        sim_time = get_sim_time(sim_ros_node)
        

        if sim_time < SIM_INIT_TIME:
            # print("cur sim time", sim_time)
            continue
        if not img_h_raw or not img_h_raw.header.stamp:
            continue
        if not img_l_raw or not img_l_raw.header.stamp:
            continue
        if not img_r_raw or not img_r_raw.header.stamp:
            continue

        #更改时间戳同步计算方式
        ts_h = img_h_raw.header.stamp.sec + img_h_raw.header.stamp.nanosec * 1e-9
        ts_l = img_l_raw.header.stamp.sec + img_l_raw.header.stamp.nanosec * 1e-9
        ts_r = img_r_raw.header.stamp.sec + img_r_raw.header.stamp.nanosec * 1e-9

        if (
            abs(ts_h - ts_l) < 0.5
            and abs(ts_h - ts_r) < 0.5
        ):
            fps_start_time = sim_time
            # 如果当前帧和上一帧时间戳相同，跳过,降低帧率，和图片保持一致
            if abs(ts_h - last_ts_h) < 1e-3:
                print("[INFO] Same frame as last time, skipping...")
                continue

            last_ts_h = ts_h
            if fps_start_time is None:
                fps_start_time = sim_time  #只初始化一次
            count += 1

            img_h = bridge.compressed_imgmsg_to_cv2(
                img_h_raw, desired_encoding="rgb8"
            )
            img_l = bridge.compressed_imgmsg_to_cv2(
                img_l_raw, desired_encoding="rgb8"
            )
            img_r = bridge.compressed_imgmsg_to_cv2(
                img_r_raw, desired_encoding="rgb8"
            )

            # img_h = simulate_base64_image(img_h)
            # img_l = simulate_base64_image(img_l)
            # img_r = simulate_base64_image(img_r)

            # resized_head = resize_img(img_h)
            # resized_left = resize_img(img_l)
            # resized_right = resize_img(img_r)
            resized_head = preprocess_image(img_h, device='cuda')
            resized_left = preprocess_image(img_l, device='cuda')
            resized_right = preprocess_image(img_r, device='cuda')

            state_tensor = torch.tensor(act_raw, dtype=torch.float32).unsqueeze(0).to('cuda')
            observation = {
                "observation.images.head": resized_head,
                "observation.images.hand_left": resized_left,
                "observation.images.hand_right": resized_right,
                "observation.state": state_tensor,
                "task": task_instruction
            }

            # print("here")
            # with torch.inference_mode():

            action = policy.select_action(observation)
            sim_ros_node.publish_joint_command_16(action)
            # input("Press Enter to continue...")

                # end_time.record()
                # torch.cuda.synchronize()  # 等待所有 CUDA 流执行完毕

                # elapsed_time_ms = start_time.elapsed_time(end_time)
                # print(f"[TIME] select_action took {elapsed_time_ms:.3f} ms")

                # if sim_ros_node.action_data is not None and sim_ros_node.action_idx < len(sim_ros_node.action_data):
                #     # 取出当前动作，转成torch tensor，并保持batch维度[1, N]
                #     print("[publish] Joint positions")

                #     action = torch.tensor(sim_ros_node.action_data[sim_ros_node.action_idx]).unsqueeze(0).float()
                #     sim_ros_node.publish_joint_command_for_test(action)
                #     sim_ros_node.action_idx += 1
                # else:
                #     # 读完了，重头开始循环或者停止
                #     sim_ros_node.action_idx = 0
                # sim_ros_node.publish_joint_command(action)

                # action = np.linspace(-np.pi / 2, np.pi / 2, 16)
                # sim_ros_node.publish_joint_command(action)

            # action = np.linspace(-np.pi / 2, np.pi / 2, 16)
            # sim_ros_node.publish_joint_command(action)

            #统计当前循环的实际执行频率
            elapsed = sim_time - fps_start_time
            if elapsed >= 60:
                fps = count / elapsed
                print(f"[INFO] Average FPS: {fps:.2f} (over {elapsed:.2f} seconds)")
                print(f"[INFO] sim_time: {sim_time:.2f} (fps_start_time {fps_start_time:.2f} )")
                fps_start_time = sim_time  # 重新起点
                count = 0

            sim_ros_node.loop_rate.sleep()


        else:
            print("====[INFO] Conditions NOT met====")
            if img_h_raw:
                print(f"Head  stamp: {img_h_raw.header.stamp.sec}.{img_h_raw.header.stamp.nanosec:09d}")
            else:
                print("Head  stamp: None")

            if img_l_raw:
                print(f"Left  stamp: {img_l_raw.header.stamp.sec}.{img_l_raw.header.stamp.nanosec:09d}")
            else:
                print("Left  stamp: None")

            if img_r_raw:
                print(f"Right stamp: {img_r_raw.header.stamp.sec}.{img_r_raw.header.stamp.nanosec:09d}")
            else:
                print("Right stamp: None")

            # if act_raw:
            #     print(f"Joint state: OK")
            # else:
            #     print("Joint state: None")
            print("===============================")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="config")
    parser.add_argument("--task_name", type=str)
    args = parser.parse_args()
    print(f"args.task_name:{args.task_name}")
    print(f"args:{args}")
    
    infer(args)



