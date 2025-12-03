from rclpy.qos import (
    QoSProfile,
    QoSHistoryPolicy,
    QoSReliabilityPolicy,
    QoSDurabilityPolicy,
)
from rclpy.node import Node
from rclpy.parameter import Parameter

from sensor_msgs.msg import (
    CompressedImage,
    JointState,
)
from collections import deque
import threading

import numpy as np

# ybg:For test
# import h5py
# import pyarrow.parquet as pq
import pandas as pd
QOS_PROFILE_LATEST = QoSProfile(
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=30,
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
)

'''
夹爪状态量
SIM                 闭合 打开
            state    0   0.8
            action   0   1
CLIP                闭合 打开
            state    0.8 0
            action   1   0          
Datasets            闭合 打开
            state   108  0
            action  0.94 0  
'''

class SimROSNode(Node):
    def __init__(self, node_name="univla_node"):
        super().__init__(
            node_name,
            parameter_overrides=[Parameter("use_sim_time", Parameter.Type.BOOL, True)],
        )

        # publish
        self.pub_joint_command = self.create_publisher(
            JointState,
            "/sim/target_joint_state",
            QOS_PROFILE_LATEST,
        )

        # subscribe
        self.sub_img_head = self.create_subscription(
            CompressedImage,
            "/sim/head_img",
            self.callback_rgb_image_head,
            1,
        )

        self.sub_img_left_wrist = self.create_subscription(
            CompressedImage,
            "/sim/left_wrist_img",
            self.callback_rgb_image_left_wrist,
            1,
        )

        self.sub_img_right_wrist = self.create_subscription(
            CompressedImage,
            "/sim/right_wrist_img",
            self.callback_rgb_image_right_wrist,
            1,
        )

        self.sub_js = self.create_subscription(
            JointState,
            "/joint_states",
            self.callback_joint_state,
            1,
        )

        self.count = 0

        # msg
        self.lock_img_head = threading.Lock()
        self.lock_img_left_wrist = threading.Lock()
        self.lock_img_right_wrist = threading.Lock()

        self.message_buffer = deque(maxlen=30)
        self.lock_joint_state = threading.Lock()
        # self.obs_joint_state = JointState()
        # self.cur_joint_state = JointState()
        self.obs_joint_state = None
        self.cur_joint_state = None
        # loop
        self.loop_rate = self.create_rate(30.0)
        # self.loop_rate = self.create_rate(5.0)

        self.img_head = None
        self.img_left_wrist = None
        self.img_right_wrist = None

        # for test
        # self.action_data = self.load_h5_actions("/root/workspace/main/AgiBot-World/checkpoints/test/aligned_joints.h5")
        # self.action_data = self.load_Parquet_actions("/root/workspace/main/AgiBot-World/checkpoints/test/episode_000011.parquet")
        # self.action_idx = 0

    # for test
    # def load_h5_actions(self, filepath):
    #     with h5py.File(filepath, "r") as f:
    #         print("Top keys:", list(f.keys()))  # ['action', 'state', 'timestamp']

    #         # 看 action 下有哪些子键
    #         print("action children:", list(f["action"].keys()))
    #         # 例如 ['joint']

    #         print("action/joint children:", list(f["action/joint"].keys()))
    #         # 例如 ['position']            

            
    #         actions = f["action/joint/position"][:]   # 读取全部动作数据，numpy数组
    #         print("Loaded action data shape:", actions.shape)
    #         return actions
        
    # def load_Parquet_actions(self, filepath):
    #     """
    #     从 parquet 文件加载动作数据

    #     Args:
    #         filepath (str): parquet 文件路径
        
    #     Returns:
    #         np.ndarray: 动作数据，shape (帧数, 动作维度)
    #     """
    #     import pyarrow.parquet as pq
    #     import numpy as np

    #     # 读取 parquet 文件为 table
    #     table = pq.read_table(filepath)

    #     # 转成 pandas DataFrame
    #     df = table.to_pandas()

    #     print("Parquet columns:", df.columns)

    #     # 假设 'action' 列是 list 类型（数组）
    #     # 取所有行 action 转成 np.ndarray
    #     # 形如 (N, M)
    #     actions_list = df["action"].tolist()

    #     # 转 np.ndarray
    #     actions_array = np.array(actions_list, dtype=np.float32)

    #     print(f"Loaded actions shape from parquet: {actions_array.shape}")
    #     print(f"\n {actions_array}")

    #     return actions_array

        
    def callback_rgb_image_head(self, msg):
        # print(msg.header)
        with self.lock_img_head:
            self.img_head = msg

    def callback_rgb_image_left_wrist(self, msg):
        # print(msg.header)
        with self.lock_img_left_wrist:
            self.img_left_wrist = msg

    def callback_rgb_image_right_wrist(self, msg):
        # print(msg.header)
        with self.lock_img_right_wrist:
            self.img_right_wrist = msg

    def get_img_head(self):
        with self.lock_img_head:
            return self.img_head

    def get_img_left_wrist(self):
        with self.lock_img_left_wrist:
            return self.img_left_wrist

    def get_img_right_wrist(self):
        with self.lock_img_right_wrist:
            return self.img_right_wrist

    def threshold_action(self, value: float, thresh: float = 0.35) -> float:
        """
        将动作值二值化，超过阈值返回 0.0（张开），否则返回 1.0（闭合）。
        """
        return 1.0 if value >= thresh else 0.0
        # return value



    # def publish_joint_command_16(self, action):
    #     """
    #     Publish joint commands to the robot.

    #     Args:
    #         action (torch.Tensor): shape [1, 16], float32
    #     """
    #     action = action.squeeze(0).cpu().numpy()  # 转到 numpy
    #     # print("[Publish OUTPUT] action shape:", action.shape)

    #     cmd_msg = JointState()
    #     cmd_msg.header.stamp = self.get_clock().now().to_msg()  # 加时间戳！

    #     cmd_msg.name = [
    #         "idx21_arm_l_joint1",
    #         "idx22_arm_l_joint2",
    #         "idx23_arm_l_joint3",
    #         "idx24_arm_l_joint4",
    #         "idx25_arm_l_joint5",
    #         "idx26_arm_l_joint6",
    #         "idx27_arm_l_joint7",
    #         "idx41_gripper_l_outer_joint1",
    #         "idx61_arm_r_joint1",
    #         "idx62_arm_r_joint2",
    #         "idx63_arm_r_joint3",
    #         "idx64_arm_r_joint4",
    #         "idx65_arm_r_joint5",
    #         "idx66_arm_r_joint6",
    #         "idx67_arm_r_joint7",
    #         "idx81_gripper_r_outer_joint1",
    #     ]

    #     cmd_msg.position = [0.0] * len(cmd_msg.name)

    #     # 左臂 7 个自由度
    #     cmd_msg.position[0] = action[2]
    #     cmd_msg.position[1] = action[3]
    #     cmd_msg.position[2] = action[4]
    #     cmd_msg.position[3] = action[5]
    #     cmd_msg.position[4] = action[6]
    #     cmd_msg.position[5] = action[7]
    #     cmd_msg.position[6] = action[8]

    #     # 左夹爪，做 clip
    #     cmd_msg.position[7] = np.clip(1.0 - action[0], 0.0, 1.0)
    #     # cmd_msg.position[7] = 0

    #     # 右臂 7 个自由度
    #     cmd_msg.position[8] = action[9]
    #     cmd_msg.position[9] = action[10]
    #     cmd_msg.position[10] = action[11]
    #     cmd_msg.position[11] = action[12]
    #     cmd_msg.position[12] = action[13]
    #     cmd_msg.position[13] = action[14]
    #     cmd_msg.position[14] = action[15]
    #     # cmd_msg.position[13] = 0

    #     # 右夹爪，做 clip
    #     cmd_msg.position[15] = np.clip(1.0 - action[1], 0.0, 1.0)

    #     print("[publish] Joint positions:", cmd_msg.position)
    #     self.pub_joint_command.publish(cmd_msg)

    def publish_joint_command_16(self, action):
        """
        Publish joint commands to the robot.

        Args:
            action (torch.Tensor): shape [1, 16], float32
        """
        action = action.squeeze(0).cpu().numpy()  # 转到 numpy
        # print("[Publish OUTPUT] action shape:", action.shape)

        cmd_msg = JointState()
        cmd_msg.header.stamp = self.get_clock().now().to_msg()  

        cmd_msg.name = [
            "idx21_arm_l_joint1",
            "idx22_arm_l_joint2",
            "idx23_arm_l_joint3",
            "idx24_arm_l_joint4",
            "idx25_arm_l_joint5",
            "idx26_arm_l_joint6",
            "idx27_arm_l_joint7",
            "idx41_gripper_l_outer_joint1",
            "idx61_arm_r_joint1",
            "idx62_arm_r_joint2",
            "idx63_arm_r_joint3",
            "idx64_arm_r_joint4",
            "idx65_arm_r_joint5",
            "idx66_arm_r_joint6",
            "idx67_arm_r_joint7",
            "idx81_gripper_r_outer_joint1",
        ]

        cmd_msg.position = [0.0] * len(cmd_msg.name)
        '''
        夹爪状态量
        SIM                 闭合 打开
                    state    0   0.8
                    action   0   1
        CLIP                闭合 打开
                    state    0.8 0
                    action   1   0          
        Datasets            闭合 打开
                    state   108  0
                    action  0.94 0  
        '''
        # 左臂
        for i in range(7):
            cmd_msg.position[i] = action[2+i]

        # 左夹爪
        cmd_msg.position[7] = np.clip(1.0 - self.threshold_action(action[0]), 0.0, 1.0)

        # 右臂
        for i in range(7):
            cmd_msg.position[8 + i] = action[9 + i]

        # 右夹爪
        cmd_msg.position[15] = np.clip(1.0 - self.threshold_action(action[1]), 0.0, 1.0)

        
        # print("[publish] Joint positions:", cmd_msg.position)
        self.pub_joint_command.publish(cmd_msg)

    # def publish_joint_command_16(self, action):
    #     """
    #     Publish joint commands to the robot.

    #     Args:
    #         action (torch.Tensor): shape [1, 16], float32
    #     """
    #     action = action.squeeze(0).cpu().numpy()  # 转到 numpy
    #     # print("[Publish OUTPUT] action shape:", action.shape)

    #     cmd_msg = JointState()
    #     cmd_msg.header.stamp = self.get_clock().now().to_msg()  

    #     cmd_msg.name = [
    #         "idx21_arm_l_joint1",
    #         "idx22_arm_l_joint2",
    #         "idx23_arm_l_joint3",
    #         "idx24_arm_l_joint4",
    #         "idx25_arm_l_joint5",
    #         "idx26_arm_l_joint6",
    #         "idx27_arm_l_joint7",
    #         "idx41_gripper_l_outer_joint1",
    #         "idx61_arm_r_joint1",
    #         "idx62_arm_r_joint2",
    #         "idx63_arm_r_joint3",
    #         "idx64_arm_r_joint4",
    #         "idx65_arm_r_joint5",
    #         "idx66_arm_r_joint6",
    #         "idx67_arm_r_joint7",
    #         "idx81_gripper_r_outer_joint1",
    #     ]

    #     cmd_msg.position = [0.0] * len(cmd_msg.name)
    #     '''
    #     夹爪状态量
    #     SIM                 闭合 打开
    #                 state    0   0.8
    #                 action   0   1
    #     CLIP                闭合 打开
    #                 state    0.8 0
    #                 action   1   0          
    #     Datasets            闭合 打开
    #                 state   108  0
    #                 action  0.94 0  
    #     '''
    #     # 左臂
    #     for i in range(7):
    #         cmd_msg.position[i] = action[2+i]

    #     # 左夹爪
    #     cmd_msg.position[7] = self.threshold_action(np.clip(1.0 - action[0], 0.0, 1.0))

    #     # 右臂
    #     for i in range(7):
    #         cmd_msg.position[8 + i] = action[9 + i]

    #     # 右夹爪
    #     cmd_msg.position[15] = self.threshold_action(np.clip(1.0 - action[1], 0.0, 1.0))

        
    #     # print("[publish] Joint positions:", cmd_msg.position)
    #     self.pub_joint_command.publish(cmd_msg)

    def publish_joint_command_for_test(self, action):
        """
        Publish joint commands to the robot.

        Args:
            action (np.ndarray): shape [14]
        """
        if hasattr(action, "cpu"):
            action = action.squeeze(0).cpu().numpy()

        cmd_msg = JointState()
        cmd_msg.header.stamp = self.get_clock().now().to_msg()

        cmd_msg.name = [
            "idx21_arm_l_joint1",
            "idx22_arm_l_joint2",
            "idx23_arm_l_joint3",
            "idx24_arm_l_joint4",
            "idx25_arm_l_joint5",
            "idx26_arm_l_joint6",
            "idx27_arm_l_joint7",
            "idx41_gripper_l_outer_joint1",
            "idx61_arm_r_joint1",
            "idx62_arm_r_joint2",
            "idx63_arm_r_joint3",
            "idx64_arm_r_joint4",
            "idx65_arm_r_joint5",
            "idx66_arm_r_joint6",
            "idx67_arm_r_joint7",
            "idx81_gripper_r_outer_joint1",
        ]

        cmd_msg.position = [0.0] * len(cmd_msg.name)

        # 左臂
        for i in range(7):
            cmd_msg.position[i] = action[i]

        # 左夹爪
        cmd_msg.position[7] = 0.0

        # 右臂
        for i in range(7):
            cmd_msg.position[8 + i] = action[7 + i]

        # 右夹爪
        cmd_msg.position[15] = 0.0

        print("[publish] Joint positions:", cmd_msg.position)
        self.pub_joint_command.publish(cmd_msg)


    # def callback_joint_state(self, msg):
    #     # print(msg.header)
    #     self.cur_joint_state = msg
    #     # print("\n##########################\n")
    #     # print(self.cur_joint_state)
    #     # print("\n##########################\n")

    #     joint_name_state_dict = {}
    #     for idx, name in enumerate(msg.name):
    #         joint_name_state_dict[name] = msg.position[idx]

    #     msg_remap = JointState()
    #     msg_remap.header = msg.header
    #     msg_remap.name = []
    #     msg_remap.velocity = []
    #     msg_remap.effort = []
    #     msg_remap.position.append(joint_name_state_dict["idx21_arm_l_joint1"])
    #     msg_remap.position.append(joint_name_state_dict["idx22_arm_l_joint2"])
    #     msg_remap.position.append(joint_name_state_dict["idx23_arm_l_joint3"])
    #     msg_remap.position.append(joint_name_state_dict["idx24_arm_l_joint4"])
    #     msg_remap.position.append(joint_name_state_dict["idx25_arm_l_joint5"])
    #     msg_remap.position.append(joint_name_state_dict["idx26_arm_l_joint6"])
    #     msg_remap.position.append(joint_name_state_dict["idx27_arm_l_joint7"])
    #     left_gripper_pos = min(1, max(0.0, (0.8 - (joint_name_state_dict["idx41_gripper_l_outer_joint1"]))))
    #     msg_remap.position.append(left_gripper_pos)

    #     msg_remap.position.append(joint_name_state_dict["idx61_arm_r_joint1"])
    #     msg_remap.position.append(joint_name_state_dict["idx62_arm_r_joint2"])
    #     msg_remap.position.append(joint_name_state_dict["idx63_arm_r_joint3"])
    #     msg_remap.position.append(joint_name_state_dict["idx64_arm_r_joint4"])
    #     msg_remap.position.append(joint_name_state_dict["idx65_arm_r_joint5"])
    #     msg_remap.position.append(joint_name_state_dict["idx66_arm_r_joint6"])
    #     msg_remap.position.append(joint_name_state_dict["idx67_arm_r_joint7"])
    #     right_gripper_pos = min(1, max(0.0, (0.8 - (joint_name_state_dict["idx81_gripper_r_outer_joint1"]))))
    #     msg_remap.position.append(right_gripper_pos)

    #     with self.lock_joint_state:
    #         self.obs_joint_state = msg_remap


    def construct_observation_state_16(self, joint_state_msg):
        obs_state = np.zeros(16, dtype=np.float32)
        joint_dict = {name: pos for name, pos in zip(joint_state_msg.name, joint_state_msg.position)}
    
        # 原始夹爪角度（未 clip）
        # raw_left = joint_dict.get("idx41_gripper_l_outer_joint1", 0.0)
        # raw_right = joint_dict.get("idx81_gripper_r_outer_joint1", 0.0)

        # 0-1 左右夹爪开度
        left_gripper_pos = min(108, max(0.0, (0.8 - joint_dict.get("idx41_gripper_l_outer_joint1", 0)) * 135))
        right_gripper_pos = min(108, max(0.0, (0.8 - joint_dict.get("idx81_gripper_r_outer_joint1", 0)) * 135))
        obs_state[0] = left_gripper_pos
        obs_state[1] = right_gripper_pos

        # === 打印调试 ===
        # print(f"[Gripper DEBUG] raw_left: {raw_left:.4f} -> clip: {left_gripper_pos:.4f}")
        # print(f"[Gripper DEBUG] raw_right: {raw_right:.4f} -> clip: {right_gripper_pos:.4f}")

        # 3-14 左右手臂关节角度，共14个
        left_arm_names = [
            "idx21_arm_l_joint1",
            "idx22_arm_l_joint2",
            "idx23_arm_l_joint3",
            "idx24_arm_l_joint4",
            "idx25_arm_l_joint5",
            "idx26_arm_l_joint6",
            "idx27_arm_l_joint7",
        ]
        right_arm_names = [
            "idx61_arm_r_joint1",
            "idx62_arm_r_joint2",
            "idx63_arm_r_joint3",
            "idx64_arm_r_joint4",
            "idx65_arm_r_joint5",
            "idx66_arm_r_joint6",
            "idx67_arm_r_joint7",
        ]

        # 左手关节cur val
        for i, name in enumerate(left_arm_names):
            obs_state[2 + i] = joint_dict.get(name, 0.0)
        # 右手关节cur val
        for i, name in enumerate(right_arm_names):
            obs_state[9 + i] = joint_dict.get(name, 0.0)

        timestamp = joint_state_msg.header.stamp.sec + joint_state_msg.header.stamp.nanosec * 1e-9

        # print(obs_state)
        # return obs_state ,timestamp
        return obs_state
    
        def construct_observation_state_55(self, joint_state_msg):
            obs_state = np.zeros(55, dtype=np.float32)
            joint_dict = {name: pos for name, pos in zip(joint_state_msg.name, joint_state_msg.position)}
        
            # 0-1 左右夹爪开度
            left_gripper_pos = min(1, max(0.0, (0.8 - joint_dict.get("idx41_gripper_l_outer_joint1", 0))))
            right_gripper_pos = min(1, max(0.0, (0.8 - joint_dict.get("idx81_gripper_r_outer_joint1", 0))))
            obs_state[0] = left_gripper_pos
            obs_state[1] = right_gripper_pos

            # # 2-9 左右四元数

            # quat = [
            #     robot_pose_msg.orientation.x,
            #     robot_pose_msg.orientation.y,
            #     robot_pose_msg.orientation.z,
            #     robot_pose_msg.orientation.w,
            # ]
            # obs_state[2:6] = quat  # left_xyzw
            # obs_state[6:10] = quat  # right_xyzw

            # # 10-15 左右xyz位置
            # pos = [
            #     robot_pose_msg.position.x,
            #     robot_pose_msg.position.y,
            #     robot_pose_msg.position.z,
            # ]
            # obs_state[10:13] = pos  # left_xyz
            # obs_state[13:16] = pos  # right_xyz

            # # 16-17 头部yaw，patch
            # obs_state[16] = head_state.get("yaw", 0.0)
            # obs_state[17] = head_state.get("patch", 0.0)

            # 18-31 左右手臂关节角度，共14个
            left_arm_names = [
                "idx21_arm_l_joint1",
                "idx22_arm_l_joint2",
                "idx23_arm_l_joint3",
                "idx24_arm_l_joint4",
                "idx25_arm_l_joint5",
                "idx26_arm_l_joint6",
                "idx27_arm_l_joint7",
            ]
            right_arm_names = [
                "idx61_arm_r_joint1",
                "idx62_arm_r_joint2",
                "idx63_arm_r_joint3",
                "idx64_arm_r_joint4",
                "idx65_arm_r_joint5",
                "idx66_arm_r_joint6",
                "idx67_arm_r_joint7",
            ]

            # 左手关节cur val
            for i, name in enumerate(left_arm_names):
                obs_state[18 + i] = joint_dict.get(name, 0.0)
            # 右手关节cur val
            for i, name in enumerate(right_arm_names):
                obs_state[25 + i] = joint_dict.get(name, 0.0)

            for i, name in enumerate(left_arm_names):
                obs_state[32 + i] = joint_dict.get(name, 0.0)
            for i, name in enumerate(right_arm_names):
                obs_state[39 + i] = joint_dict.get(name, 0.0)

            return obs_state
    
    def callback_joint_state(self, msg):
        # print(msg.header)
        self.cur_joint_state = msg
        msg_remap = self.construct_observation_state_16(self.cur_joint_state)

        with self.lock_joint_state:
            self.obs_joint_state = msg_remap


    
    def get_joint_state(self):
        with self.lock_joint_state:
            return self.obs_joint_state

