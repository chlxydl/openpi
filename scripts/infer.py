#!/usr/bin/env python3

import time
from array import array
from collections import deque

import cv2
import numpy as np
import rclpy
import torch
from cv_bridge import CvBridge
from geometry_msgs.msg import TransformStamped
# from lerobot.common.policies.act.modeling_act import ACTPolicy
# from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
# from rclpy.publisher import Publisher
from robot_interfaces.msg import JointPositionCommands
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import Image, JointState
# from tf2_ros import TransformException
# from tf2_ros.buffer import Buffer
# from tf2_ros.transform_listener import TransformListener


class ArmControl(Node):
    def __init__(self):
        super().__init__('arm_control')

        # Load parameters
        self.load_parameters()

        # Init the policy
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        # self.policy = SmolVLAPolicy.from_pretrained(self.policy_path)
        args = serve_policy.Args(   
            port=8000,
            record=False,
        )
        args.policy = serve_policy.Checkpoint(
            config="pi0_xxl_vla",
            dir="/work/outputs/debug/pi0_xxl_vla/pi0_xxl_test/10000/params",
            # dir="ckpt"
        )
        self.policy = serve_policy.create_policy(args)

        self.img_size = (self.img_width, self.img_height)

        # Init subscriptions for states
        self.joint_state_sub = Subscriber(
            self, JointState, '/joint/states')

        # Init subscriptions for images
        self.env_image_sub = Subscriber(self, Image, self.env_image_topic)
        self.head_image_sub = Subscriber(self, Image, self.head_image_topic)
        self.right_wrist_image_sub = Subscriber(
            self, Image, self.right_wrist_image_topic)
        self.env_image = np.zeros(
            (self.img_height, self.img_width, 3), np.uint8)
        self.head_image = np.zeros(
            (self.img_height, self.img_width, 3), np.uint8)
        self.right_wrist_image = np.zeros(
            (self.img_height, self.img_width, 3), np.uint8)
        self.bridge = CvBridge()

        # Create a time synchronizer
        self.ts = ApproximateTimeSynchronizer(
            [self.joint_state_sub, self.env_image_sub, self.head_image_sub,
             self.right_wrist_image_sub], queue_size=10,
            slop=self.max_delay)
        self.ts.registerCallback(self.sync_callback)

        # Init variables to save the historical data
        self.env_img_buffer = deque(maxlen=self.history_size)
        self.head_img_buffer = deque(maxlen=self.history_size)
        self.right_wrist_img_buffer = deque(maxlen=self.history_size)
        self.right_arm_joints_buffer = deque(maxlen=self.history_size)

        # Init the policy dict
        if self.using_env_image:
            self.observation = {
                "observation.right_arm_joints": None,
                "observation.images.cam1": None,
                "observation.images.cam2": None,
                "observation.images.cam3": None,
                "task": [self.task_description],
            }
        else:
            self.observation = {
                "observation.right_arm_joints": None,
                "observation.images.cam2": None,
                "observation.images.cam3": None,
                "task": [self.task_description],
            }

        # Init the joint position command publisher
        self.joint_position_cmd_pub = self.create_publisher(
            JointPositionCommands, '/joint/test_arm_head_waist_commands', 10)
        self.joint_position_cmd = JointPositionCommands()
        self.joint_position_cmd.joint_positions = array(
            'f', self.init_positions)

        # Timer for control
        self.timer = self.create_timer(
            1.0 / self.timer_freq, self.control_callback)

    def sync_callback(self, joint_state_msg, env_img_msg, head_img_msg,
                      right_wrist_img_msg):
        right_arm_joints = joint_state_msg.position[15:22]
        right_finger_positions = joint_state_msg.position[33:37]
        # Retrieve the gripper state based on the positions of ring,
        # middle, ring fingers, and thumb flex
        if np.linalg.norm(right_finger_positions) > self.gripper_state_threshold:
            right_arm_joints = list(right_arm_joints) + [0.0]
        else:
            right_arm_joints = list(right_arm_joints) + [1.0]

        # Save variables to the buffer
        self.right_arm_joints_buffer.append(
            torch.tensor(right_arm_joints, dtype=torch.float32))
        self.env_img_buffer.append(self.process_img_msg(env_img_msg))
        self.head_img_buffer.append(self.process_img_msg(head_img_msg))
        self.right_wrist_img_buffer.append(
            self.process_img_msg(right_wrist_img_msg))

        self.get_logger().info("received data")

    def control_callback(self):
        """
        Timer callback for control loop.
        """
        # Check if we have enough data in buffers
        if (len(self.env_img_buffer) >= self.history_size and
            len(self.head_img_buffer) >= self.history_size and
            len(self.right_wrist_img_buffer) >= self.history_size and
                len(self.right_arm_joints_buffer) >= self.history_size):
            self.get_logger().info("Running the policy")
            self.run()

    def run(self):
        # Stack tensors
        right_arm_joints_stack = torch.stack(
            list(self.right_arm_joints_buffer), dim=0).to(self.device)
        env_img_stack = torch.stack(
            list(self.env_img_buffer), dim=0).to(self.device)
        head_img_stack = torch.stack(
            list(self.head_img_buffer), dim=0).to(self.device)
        right_wrist_img_stack = torch.stack(
            list(self.right_wrist_img_buffer), dim=0).to(self.device)

        # Create the policy input dict
        # self.observation["observation.right_arm_joints"] = right_arm_joints_stack
        # if self.using_env_image:
        #     self.observation["observation.images.cam1"] = env_img_stack
        # self.observation["observation.images.cam2"] = head_img_stack
        # self.observation["observation.images.cam3"] = right_wrist_img_stack
        observation = {
            'state': right_arm_joints_stack, # 8
            'observation.images.head': head_img_stack, # 0-1
            'observation.images.hand_right': right_wrist_img_stack,
            # left_wrist_img_stack?
            'observation.images.hand_left': torch.zeros((3, 224, 224), dtype=torch.float32),
            'prompt': "Place the bottle to the pad."
        }

        # Get the action from the policy
        with torch.inference_mode():
            actions = self.policy.infer(observation) # 50 * 8
            self.get_logger().info(f"Actions: {actions}")

        # Publish the action
        self.publish_actions(actions)

    def publish_actions(self, actions):
        """
        Publish the actions sequentially.
        """
        for i in range(10):
            action = actions[i]
            self.publish_action(action)
            time.sleep(0.03)  # Sleep for a short duration between actions

    def publish_action(self, action):
        """
        Publish the action.
        """
        action = action.squeeze(0).to("cpu").numpy()
        self.joint_position_cmd.joint_positions[3:10] = array("f", action[:7])
        if action[-1] > 0.5:
            # TODO: Check the correct values
            self.joint_position_cmd.joint_positions[20] = 500.0
            self.joint_position_cmd.joint_positions[21] = 500.0
            self.joint_position_cmd.joint_positions[22] = 500.0
            self.joint_position_cmd.joint_positions[23] = 500.0
            self.joint_position_cmd.joint_positions[24] = 500.0
            self.joint_position_cmd.joint_positions[25] = 500.0
        else:
            self.joint_position_cmd.joint_positions[20] = 1000.0
            self.joint_position_cmd.joint_positions[21] = 1000.0
            self.joint_position_cmd.joint_positions[22] = 1000.0
            self.joint_position_cmd.joint_positions[23] = 1000.0
            self.joint_position_cmd.joint_positions[24] = 1000.0
            self.joint_position_cmd.joint_positions[25] = 1000.0

        self.joint_position_cmd_pub.publish(self.joint_position_cmd)

    def process_img_msg(self, img_msg):
        """
        Process the image message and return a tensor.

        Parameters
        ----------
        img_msg : Image

        Returns
        -------
        torch.Tensor
            The processed image tensor.
        """
        img = cv2.resize(self.bridge.imgmsg_to_cv2(img_msg, "bgr8"),
                         self.img_size, interpolation=cv2.INTER_LINEAR)
        return torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0

    def process_joint_state_msg(self, trans: TransformStamped,
                                finger_positions: np.ndarray):
        """
        Process the joint state message and return a tensor.
        """
        # Retrieve the pose of the right wrist roll
        self.right_ee_state_sixd[:3] = [trans.transform.translation.x,
                                        trans.transform.translation.y,
                                        trans.transform.translation.z]
        self.right_ee_state_sixd[3:9] = so3_to_sixd([
            trans.transform.rotation.x,
            trans.transform.rotation.y,
            trans.transform.rotation.z,
            trans.transform.rotation.w
        ])

        # Retrieve the gripper state based on the positions of ring,
        # middle, ring fingers, and thumb flex
        if np.linalg.norm(finger_positions) > self.gripper_state_threshold:
            self.right_ee_state_sixd[9] = 0
        else:
            self.right_ee_state_sixd[9] = 1

    def load_parameters(self):
        self.declare_parameter("img_height", rclpy.Parameter.Type.INTEGER)
        self.declare_parameter("img_width", rclpy.Parameter.Type.INTEGER)
        self.declare_parameter("timer_freq", rclpy.Parameter.Type.INTEGER)
        self.declare_parameter("max_delay", rclpy.Parameter.Type.DOUBLE)
        self.declare_parameter("history_size", rclpy.Parameter.Type.INTEGER)
        self.declare_parameter(
            "gripper_state_threshold", rclpy.Parameter.Type.DOUBLE)
        self.declare_parameter("policy_path", rclpy.Parameter.Type.STRING)
        self.declare_parameter("task_description", rclpy.Parameter.Type.STRING)
        self.declare_parameter("using_env_image", rclpy.Parameter.Type.BOOL)
        self.declare_parameter("env_image_topic", rclpy.Parameter.Type.STRING)
        self.declare_parameter("head_image_topic", rclpy.Parameter.Type.STRING)
        self.declare_parameter("right_wrist_image_topic",
                               rclpy.Parameter.Type.STRING)

        self.img_height = self.get_parameter(
            "img_height").get_parameter_value().integer_value
        self.img_width = self.get_parameter(
            "img_width").get_parameter_value().integer_value
        self.timer_freq = self.get_parameter(
            "timer_freq").get_parameter_value().integer_value
        self.max_delay = self.get_parameter(
            "max_delay").get_parameter_value().double_value
        self.history_size = self.get_parameter(
            "history_size").get_parameter_value().integer_value
        self.gripper_state_threshold = self.get_parameter(
            "gripper_state_threshold").get_parameter_value().double_value
        self.policy_path = self.get_parameter(
            "policy_path").get_parameter_value().string_value
        self.task_description = self.get_parameter(
            "task_description").get_parameter_value().string_value
        self.using_env_image = self.get_parameter(
            "using_env_image").get_parameter_value().bool_value
        self.env_image_topic = self.get_parameter(
            "env_image_topic").get_parameter_value().string_value
        self.head_image_topic = self.get_parameter(
            "head_image_topic").get_parameter_value().string_value
        self.right_wrist_image_topic = self.get_parameter(
            "right_wrist_image_topic").get_parameter_value().string_value


def main():
    rclpy.init()
    arm_control = ArmControl()
    try:
        rclpy.spin(arm_control)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


def so3_to_sixd(q):
    """
    Convert 4D quaternion (x,y,z,w) to 6D representation.
    """
    R = quaternion_to_matrix(q)
    return R[:, :2].T.reshape(-1)


def sixd_to_so3(sixd):
    """
    Convert 6D vector back to 3x3 rotation matrix using Gram-Schmidt.
    """
    a1 = sixd[:3]
    a2 = sixd[3:]

    b1 = normalize(a1)
    proj = np.dot(b1, a2) * b1
    b2 = normalize(a2 - proj)
    b3 = np.cross(b1, b2)

    R = np.stack([b1, b2, b3], axis=1)

    quat = matrix_to_quaternion(R)
    return quat


def quaternion_to_matrix(q):
    """
    Convert quaternion (x, y, z, w) to 3x3 rotation matrix.
    """
    rot = R.from_quat([q[0], q[1], q[2], q[3]])  # (x, y, z, w)
    r_mat = rot.as_matrix()
    return r_mat


def matrix_to_quaternion(rot_matrix):
    r = R.from_matrix(rot_matrix)
    quat = r.as_quat()
    return quat


def normalize(v):
    return v / (np.linalg.norm(v) + 1e-8)


if __name__ == '__main__':
    main()







##########################
import rclpy
from cv_bridge import CvBridge
import argparse
import threading

import cv2
import numpy as np
import os
import sys
import os
import sys

# 添加项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

# 添加 openpi-client 的 src 路径
OPENPI_CLIENT_PATH = os.path.join(PROJECT_ROOT, "packages", "openpi-client", "src")
sys.path.append(OPENPI_CLIENT_PATH)

# 添加 openpi 的 src 路径
OPENPI_PATH = os.path.join(PROJECT_ROOT, "src")
sys.path.append(OPENPI_PATH)

print(PROJECT_ROOT)
print(OPENPI_PATH)

# from  serve_policy  import create_policy, Args
import  serve_policy 
# 添加项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

# 添加 openpi-client 的 src 路径
OPENPI_CLIENT_PATH = os.path.join(PROJECT_ROOT, "packages", "openpi-client", "src")
sys.path.append(OPENPI_CLIENT_PATH)

# 添加 openpi 的 src 路径
OPENPI_PATH = os.path.join(PROJECT_ROOT, "src")
sys.path.append(OPENPI_PATH)

print(PROJECT_ROOT)
print(OPENPI_PATH)
# from genie_sim_ros_lerobot import SimROSNode
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from genie_sim_ros import SimROSNode
from genie_sim_ros_lerobot import SimROSNode

import time
from openpi_client import image_tools
from openpi_client import websocket_client_policy


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
        lang = ""
        raise ValueError("task does not exist")

    return lang


def get_sim_time(sim_ros_node):
    sim_time = sim_ros_node.get_clock().now().nanoseconds * 1e-9
    return sim_time

def infer(cfg):
    # client = make_client(host="GCRAZGDL1528.westus3.cloudapp.azure.com", port=8000)

    # print("policy",policy)


    observation = {
        "observation.images.head": image_tools.convert_to_uint8(
            image_tools.resize_with_pad(resized_head, 224, 224)
        ),
        "observation.images.hand_left": image_tools.convert_to_uint8(
            image_tools.resize_with_pad(resized_left, 224, 224)
        ),
        "observation.images.hand_right": image_tools.convert_to_uint8(
            image_tools.resize_with_pad(resized_right, 224, 224)
        ),
        "state": act_raw,
        "prompt": task_instruction,
    }
    
    action = policy.infer(observation)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="config")
    parser.add_argument("--task_name", type=str)
    args = parser.parse_args()
    print(f"args.task_name:{args.task_name}")
    print(f"args:{args}")
    
    infer(args)