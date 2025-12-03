import rclpy
from rclpy.qos import (
    QoSProfile,
    QoSHistoryPolicy,
    QoSReliabilityPolicy,
    QoSDurabilityPolicy,
)
from rclpy.node import Node
from sensor_msgs.msg import JointState
from pynput import keyboard
import threading

QOS_PROFILE_LATEST = QoSProfile(
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=30,
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
)
class JointAngleController(Node):
    def __init__(self):
        super().__init__('joint_angle_controller')
        self.joint_pub = self.create_publisher(JointState, "/sim/target_joint_state", QOS_PROFILE_LATEST)
        self.timer = self.create_timer(0.1, self.publish_joint_state)

        self.joint_names = [
            "idx21_arm_l_joint1", "idx22_arm_l_joint2", "idx23_arm_l_joint3", "idx24_arm_l_joint4",
            "idx25_arm_l_joint5", "idx26_arm_l_joint6", "idx27_arm_l_joint7", "idx41_gripper_l_outer_joint1",
            "idx61_arm_r_joint1", "idx62_arm_r_joint2", "idx63_arm_r_joint3", "idx64_arm_r_joint4",
            "idx65_arm_r_joint5", "idx66_arm_r_joint6", "idx67_arm_r_joint7", "idx81_gripper_r_outer_joint1"
        ]
        self.joint_angles = [0.0] * 16
        self.step = 0.05
        self.current_joint = 0  # 当前选中的关节索引，默认第0个

        self.listener_thread = threading.Thread(target=self.keyboard_listener, daemon=True)
        self.listener_thread.start()

    def publish_joint_state(self):
        msg = JointState()
        msg.name = self.joint_names
        msg.position = self.joint_angles
        msg.header.stamp = self.get_clock().now().to_msg()
        self.joint_pub.publish(msg)

        print("="*50)
        print(f"Current selected joint: {self.joint_names[self.current_joint]} (index {self.current_joint})")
        for i, (name, angle) in enumerate(zip(self.joint_names, self.joint_angles)):
            selector = "<--" if i == self.current_joint else ""
            print(f"{name:<30}: {angle:+.3f} {selector}")
        print("="*50)

    def keyboard_listener(self):
        joint_key_map = {
            '1': 0, '2': 1, '3': 2, '4': 3,
            '5': 4, '6': 5, '7': 6, '8': 7,
            '9': 8, 'q': 9, 'w': 10, 'e': 11,
            'r': 12, 't': 13, 'y': 14, 'u': 15
        }

        def on_press(key):
            try:
                if hasattr(key, 'char') and key.char in joint_key_map:
                    self.current_joint = joint_key_map[key.char]
                    print(f"Switched control to joint {self.current_joint}: {self.joint_names[self.current_joint]}")
                elif hasattr(key, 'char') and key.char == 'z':  # reset所有关节
                    self.joint_angles = [0.0] * 16
                    print("Reset all joint angles to 0.")
            except Exception:
                pass

            # 方向键控制选中关节角度增减
            if key == keyboard.Key.up:
                self.joint_angles[self.current_joint] += self.step
            elif key == keyboard.Key.down:
                self.joint_angles[self.current_joint] -= self.step

        with keyboard.Listener(on_press=on_press) as listener:
            listener.join()

def main(args=None):
    rclpy.init(args=args)
    node = JointAngleController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
