import os
import sys
from pathlib import Path
from socketio import Client
import base64
# 保证可以找到 genie_sim_ros 和 lerobot
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent / 'src'))

import threading

import cv2
import numpy as np
import torch
import rclpy
from cv_bridge import CvBridge

from dataclasses import dataclass

from genie_sim_ros import SimROSNode

# def resize_img(img, width, height):
#     resized_img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
#     resized_img = np.array(resized_img)
#     return resized_img

#改输入为模型需要格式
def resize_img(img, target_width=640, target_height=480, device='cuda'):
    """
    Resize image to fit in (target_width x target_height) while keeping aspect ratio,
    and pad with black borders if needed.
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
    # chw_img = canvas.transpose(2, 0, 1)
    # chw_img = torch.from_numpy(chw_img).float().unsqueeze(0).to(device)
    return canvas

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

class ClientPolicy:
    def __init__(self, url):
        self.url = url
        self.sio = Client()
        self.sio.connect(self.url)
        print(f"Connected to server at {self.url}")

    def reset(self):
        pass

    def np_img_array_to_base64(self, img_array):
        """
        Convert a numpy image array to a base64 string.
        img_array: numpy array of the image
        """
        # print(img_array.shape)
        _, buffer = cv2.imencode('.jpg', img_array)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        return img_base64

    def select_action(self, observation):
        """
        Select an action based on the observation.
        observation: dict containing images and state tensor
        """
        # Convert observation to a format suitable for the server

        data = {
            "image_head": self.np_img_array_to_base64(observation["observation.images.head"]),
            "image_hand_left": self.np_img_array_to_base64(observation["observation.images.hand_left"]),
            "image_hand_right": self.np_img_array_to_base64(observation["observation.images.hand_right"]),
            "state": observation["observation.state"].tolist(),
            "command": observation["task"] # Replace with the actual task name
        }

        # Send data to the server and get the action
        # print(data)
        response = self.sio.call('select_action', data)

        print("Server response:", response)
        action = response["action"]
        # print(np.array(action).shape)  # 输出: (3,)
        action_tensor = torch.tensor(action, dtype=torch.float32)
        return action_tensor
        # return 0
    
    def __getattr__(self, name):
        return 'client_policy'

def make_policy():

    # policy = ClientPolicy(url='http://GCRAZGDL1761.westus3.cloudapp.azure.com:8080')
    policy = ClientPolicy(url='http://GCRAZGDL1528.westus3.cloudapp.azure.com:8080')
    print("policy reset")
    policy.reset()

    return policy

def infer():
    rclpy.init()
    current_path = os.getcwd()
    sim_ros_node = SimROSNode()
    spin_thread = threading.Thread(target=rclpy.spin, args=(sim_ros_node,))
    spin_thread.start()

    bridge = CvBridge()
    count = 0
    SIM_INIT_TIME = 1
    last_ts_h = -1.0


    fps_start_time = None
    print("start loop")
    policy = make_policy()
    while rclpy.ok():
        img_h_raw = sim_ros_node.get_img_head()
        img_l_raw = sim_ros_node.get_img_left_wrist()
        img_r_raw = sim_ros_node.get_img_right_wrist()
        result = sim_ros_node.get_joint_state()
        # print("Joint state result:", result)

        # act_raw, ts_a = sim_ros_node.get_joint_state()
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
            # and abs(ts_h - ts_a) < 0.5
        ):
            # 如果当前帧和上一帧时间戳相同，跳过,降低帧率，和图片保持一致
            if abs(ts_h - last_ts_h) < 1e-3:
                print("[INFO] Same frame as last time, skipping...")
                continue

            # 记录当前处理过的帧时间戳
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

            # resized_head = resize_img(img_h)
            # resized_left = resize_img(img_l)
            # resized_right = resize_img(img_r)

            state_tensor = np.array(act_raw.position)
            observation = {
                "observation.images.head": img_h,
                "observation.images.hand_left": img_l,
                "observation.images.hand_right": img_r,
                "observation.state": state_tensor,
                "task": "iros_pack_in_the_supermarket"
                # "task": "stamp_the_seal |"
                # "task": "Pick up the stamp from the ink pad on the table with the right arm.;Stamp the document on the table with the stamp in the right arm.;Place the stamp into the ink pad on the table with the right arm."
            }
            # action = policy.select_action(observation)
            # sim_ros_node.publish_joint_command_16(action)

            action = policy.select_action(observation)
            sim_ros_node.publish_joint_command(action)
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

            print("===============================")

@dataclass
class GenerateConfig:
    task_name: str = "iros_stamp_the_seal"

if __name__ == "__main__":
    # policy, cfg = get_policy()
    # infer(policy, cfg)
    # while(1):
    #     print(cfg)
    infer()
    # while(1):
    #     print("loop")


