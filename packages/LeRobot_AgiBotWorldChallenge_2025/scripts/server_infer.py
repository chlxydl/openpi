


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

import draccus
from dataclasses import dataclass

from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.pi0.modeling_pi0 import PI0Policy
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import io
import argparse
import base64
import cv2
from flask_socketio import SocketIO, emit
def make_policy():

    # path_ckpt = "/home/v-yuboguo/lerobot/outputs/train/pi0_base_pack_in_the_supermarket/checkpoints/060000/pretrained_model" #dp
    # path_ckpt = "/home/v-yuboguo/lerobot/outputs/train/act_pack_in_the_supermarket_oral/checkpoints/100000/pretrained_model" #dp
    # path_ckpt = "/home/v-yuboguo/lerobot/outputs/train/act_pack_in_the_supermarket_oral/checkpoints/100000/pretrained_model" #dp

    # path_ckpt = "/home/v-yuboguo/lerobot/outputs/train/smolvla_pack_in_the_supermarket/checkpoints/060000/pretrained_model" #dp
    # path_ckpt = "/home/v-yuboguo/lerobot/outputs/train/pi0_base_pack_all_task/checkpoints/080000/pretrained_model" 
    path_ckpt = "/home/v-yuboguo/lerobot/outputs/train/pi0_base_pack_all_task_lang_changed/checkpoints/080000/pretrained_model" 

    print("path_ckpt:",path_ckpt)
    print("Loading policy...")
    # policy = DiffusionPolicy.from_pretrained(pretrained_name_or_path = path_ckpt, local_files_only=True)
    # policy = ACTPolicy.from_pretrained(pretrained_name_or_path = path_ckpt, local_files_only=True)
    policy = PI0Policy.from_pretrained(pretrained_name_or_path = path_ckpt, local_files_only=True)
    # policy = SmolVLAPolicy.from_pretrained(pretrained_name_or_path = path_ckpt, local_files_only=True)

    print("policy reset")
    policy.reset()

    return policy

policy = make_policy()
    
def parse_args():
    parser = argparse.ArgumentParser(description='VLNCE 4-class server')
    # parser.add_argument('--config', type=str, default='/work/configs/250506_vlnce_4cls_benchmark.yaml', help='Path to the config file')
    # parser.add_argument('--model_path', type=str, default='/work/outputs/epoch-02-loss=0.2527.pt', help='Path to the model checkpoint')
    return parser.parse_args()

args = parse_args()
def preprocess_img(img, device='cuda'):
    chw_img = img.transpose(2, 0, 1)
    chw_img = torch.from_numpy(chw_img / 255.0).float().unsqueeze(0).to(device)
    return chw_img

def preprocess_state(state, device='cuda'):
    state_tensor = torch.tensor(state, dtype=torch.float32).to(device)
    return state_tensor

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*") 
########################################
@socketio.on('select_action')
def process(data):
    state = data['state']
    command = data['command']

    try:
        # timestamp = time.strftime("%Y%m%d-%H%M%S")

        # 解码 head image
        head_img_data = base64.b64decode(data['image_head'])
        head_img_array = np.frombuffer(head_img_data, np.uint8)
        head_img = cv2.imdecode(head_img_array, cv2.IMREAD_COLOR)

        # 解码 left image
        left_img_data = base64.b64decode(data['image_hand_left'])
        left_img_array = np.frombuffer(left_img_data, np.uint8)
        left_img = cv2.imdecode(left_img_array, cv2.IMREAD_COLOR)

        # 解码 right image
        right_img_data = base64.b64decode(data['image_hand_right'])
        right_img_array = np.frombuffer(right_img_data, np.uint8)
        right_img = cv2.imdecode(right_img_array, cv2.IMREAD_COLOR)


        processed_head = preprocess_img(head_img)
        processed_left = preprocess_img(left_img)
        processed_right = preprocess_img(right_img)
        processed_state = preprocess_state(state)
        print("\nprocessed_head.shape:", processed_head.shape)
        print("\nprocessed_left.shape:", processed_left.shape)
        print("\nprocessed_right.shape:", processed_right.shape)
        print("\nprocessed_state.shape:", processed_state.shape)
        observation = {
            "observation.images.head": processed_head,
            "observation.images.hand_left": processed_left,
            "observation.images.hand_right": processed_right,
            "observation.state": processed_state,
            "task": command
        }

        print("task:", command)
        # TODO: 后续执行策略动作
        action = policy.select_action(observation)
        print("action queue length:", len(policy._action_queue))
        print("action.shape:", action.shape)
        # return {"response": "Processed successfully"}
        # return action.cpu().numpy().tolist()
        return {"action": action.cpu().numpy().tolist()}

    except Exception as e:
        import traceback
        traceback.print_exc()   # ✅ 打印完整报错堆栈
        print("[Exception]", str(e))
        return {"response": "Server error: " + str(e)}

@app.route('/')
def index():
    return "VLNCE 4-class server is running!"

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=8080, debug=False)
