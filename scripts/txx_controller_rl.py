import time
import os, sys
import math
cur_work_path = os.getcwd()
sys.path.append(cur_work_path)
import mujoco.viewer
import mujoco
import numpy as np
# from legged_gym import LEGGED_GYM_ROOT_DIR
import torch
import yaml
import collections
import copy
from filterpy.kalman import KalmanFilter
# import pygame
from threading import Thread
import keyboard
from sshkeyboard import listen_keyboard
from pynput.keyboard import Listener, Key
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import Imu
from sensor_msgs.msg import JointState
from threading import Thread
import random
import shutil
from torch.utils.tensorboard import SummaryWriter
import transforms3d


def get_gravity_orientation(quaternion):
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)

    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation

import onnxruntime as ort
def load_onnx_policy(path):
    model = ort.InferenceSession(path)
    def run_inference(input_tensor):
        ort_inputs = {model.get_inputs()[0].name: input_tensor.cpu().numpy()}
        ort_outs = model.run(None, ort_inputs)
        return torch.tensor(ort_outs[0], device="cpu")
    return run_inference

class Controller:

    def __init__(self):

        log_dir = 'curve/real'
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)
        self.writer = SummaryWriter(log_dir=log_dir)  # 创建tensorboard写入器

        ros_joint_names = ["left_ankle_pitch_joint",
                           "left_ankle_roll_joint",
                           "right_hip_yaw_joint",
                           "right_hip_pitch_joint",
                           "right_ankle_roll_joint",
                           "WAIST_R",
                           "left_hip_yaw_joint",
                           "left_hip_pitch_joint",
                           "right_knee_joint",
                           "R_SHOULDER_P",
                           "L_SHOULDER_P",
                           "L_WRIST_Y",
                           "left_knee_joint",
                           "L_SHOULDER_Y",
                           "R_SHOULDER_Y",
                           "left_hip_roll_joint",
                           "waist_yaw_joint",
                           "L_WRIST_P",
                           "WAIST_P",
                           "L_WRIST_R",
                           "R_SHOULDER_R",
                           "right_hip_roll_joint",
                           "right_ankle_pitch_joint",
                           "L_ELBOW_Y",
                           "L_SHOULDER_R",
                           "R_WRIST_R",
                           "R_WRIST_Y",
                           "R_WRIST_P",
                           "R_ELBOW_Y"]
        
        joint_names = ['left_hip_roll_joint', 'left_hip_yaw_joint', 'left_hip_pitch_joint', 'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint',
                       'right_hip_roll_joint', 'right_hip_yaw_joint', 'right_hip_pitch_joint', 'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint']

        # joint_names = ['left_hip_roll_joint', 'right_hip_roll_joint', 'left_hip_yaw_joint', 'right_hip_yaw_joint', 'left_hip_pitch_joint', 'right_hip_pitch_joint',
        #                'left_knee_joint', 'right_knee_joint', 'left_ankle_pitch_joint', 'right_ankle_pitch_joint', 'left_ankle_roll_joint', 'right_ankle_roll_joint']

        # self.kp = [200, 200, 200, 200, 200, 200, 300, 300, 40, 40, 40, 40]
        # self.kd = [2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 4, 4, 2, 2, 2, 2]
        self.kp = [200, 200, 200, 300, 40, 40, 200, 200, 200, 300,40, 40]
        self.kd = [2.5, 2.5, 2.5, 4, 2, 2, 2.5, 2.5, 2.5, 4, 2, 2]
        self.joint_indices = np.zeros(12, dtype=np.int32)
        for index, joint_name in enumerate(joint_names):
            if index < 12:
                self.joint_indices[index] = ros_joint_names.index(joint_name)

        ###################### config ###############################
        self.ang_vel_scale = 1.0
        self.dof_pos_scale = 1.0
        self.dof_vel_scale = 1.0
        self.action_scale = 0.5
        self.cmd_scale = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        self.num_actions = 12
        self.single_num_obs = 45
        self.obs_history_len = 1
        self.num_obs = self.obs_history_len * self.single_num_obs
        self.default_angles = [0.0, 0.0, -0.1, -0.25, 0.17, 0.0,
                               0.0, 0.0, -0.1, -0.25, 0.17, 0.0]
        self.target_dof_pos = np.array(self.default_angles, dtype=np.float32)
        self.n_joints = 12

        hz = 50
        self.control_dt = 1 / hz
        self.hz = hz

        self.walk_policy_path = f'/home/speedbot/dev/speedbot-v1_3/deploy/model/txx/policy_1.pt'
        self.walk_policy = torch.jit.load(self.walk_policy_path)
        self.policy = self.walk_policy

        self.omega_history_len = 10
        self.omega_history = collections.deque(maxlen=self.omega_history_len)
        
        for _ in range(self.omega_history_len):
            self.omega_history.append(np.zeros(3, dtype=np.float32))

        self.cmd = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.control_counter = 0
        self.walk_control_begin_time = 0

        self.key_listener = Listener(on_press=self._on_key_press, on_release=self._on_key_release)
        self.key_listener.start()

        self.qj = np.zeros(self.n_joints, dtype=np.float32)
        self.dqj = np.zeros(self.n_joints, dtype=np.float32)
        self.last_dqj = np.zeros(self.n_joints, dtype=np.float32)
        self.ddqj = np.zeros(self.n_joints, dtype=np.float32)
        self.effort = np.zeros(self.n_joints, dtype=np.float32)

        self.quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.omega = np.zeros(3, dtype=np.float32)
        self.action = np.zeros(self.num_actions, dtype=np.float32)

        self.quat_filter = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.omega_filter = np.zeros(3, dtype=np.float32)

        self.obs_history = collections.deque(maxlen=self.obs_history_len)
        for _ in range(self.obs_history_len):
            self.obs_history.append(np.zeros(self.single_num_obs, dtype=np.float32))
        self.obs = np.zeros(self.num_obs, dtype=np.float32)


        rclpy.init()
        node = Node("controller")
        self.imu_data_sub = node.create_subscription(Imu, '/imu/data', self.imu_data_listener_callback, 10)
        self.joint_states_sub = node.create_subscription(JointState, '/joint_states',
                                                         self.joint_states_listener_callback, 10)
        self.control_data_pub = node.create_publisher(Float64MultiArray, '/rl_motion_control_command', 10)
        
        self.state_data_sub_thread = Thread(target=rclpy.spin, args=(node,), daemon=True)
        self.state_data_sub_thread.start()

        self.is_run = False
        self.is_over = False

        self.imu_data_listen_last_time = time.time()
        self.imu_data_listen_counter = 0

        self.joint_states_listen_last_time = time.time()
        self.joint_states_listen_counter = 0

        self.gravity_orientation = np.array([0.0, 0.0, -1.0])



        self.policy_count = 0

    def imu_data_listener_callback(self, msg):
        self.omega = np.array([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z])
        self.quat = np.array([msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z])
        self.omega_history.append(self.omega)

    def joint_states_listener_callback(self, msg):
        self.qj = np.array(msg.position)[self.joint_indices]
        self.dqj = np.array(msg.velocity)[self.joint_indices]

    def control_data_pubisher_call(self, target_dof_pos):
        up_joints = [0.0]*17

        msg = Float64MultiArray()
        msg.data = (target_dof_pos.tolist()+up_joints)
        controller.control_data_pub.publish(msg)

    def _on_key_press(self, key):
        # print(key)
        if key == Key.up:  # 上方向键
            self.cmd = np.array([0.5, 0., 0.], dtype=np.float32)
        elif key == Key.down:  # 下方向键
            self.cmd = np.array([-0.3, 0., 0.], dtype=np.float32)
        elif key == Key.left:  # 左方向键
            self.cmd = np.array([0.0, -0.3, 0.], dtype=np.float32)
        elif key == Key.right:  # 右方向键
            self.cmd = np.array([0.0, 0.3, 0.], dtype=np.float32)
        elif hasattr(key, 'char') and key.char.lower() == '[':
            self.cmd[0] -= 0.1
        elif hasattr(key, 'char') and key.char.lower() == ']':
            self.cmd[0] += 0.1
        elif hasattr(key, 'char') and key.char.lower() == '5':
            self.cmd[2] += 0.1
        elif hasattr(key, 'char') and key.char.lower() == '6':
            self.cmd[2] -= 0.1
        elif hasattr(key, 'char') and key.char.lower() == '0':
            self.cmd[0] = 0
            self.cmd[1] = 0
            self.cmd[2] = 0
        elif hasattr(key, 'char') and key.char.lower() == '9':
            self.is_run = True
            self.walk_control_begin_time = time.time()
            print("run!!!")
        elif hasattr(key, 'char') and key.char.lower() == 'q':
            self.is_over = True

    def _on_key_release(self, key):
        pass

    def compute_observation(self):
        """Compute the observation vector from current state"""

        # Scale the values
        qj_scaled = (self.qj - self.default_angles) * self.dof_pos_scale
        dqj_scaled = self.dqj * self.dof_vel_scale
        self.gravity_orientation = get_gravity_orientation(self.quat)
        omega_scaled = np.mean(self.omega_history, axis=0) * self.ang_vel_scale

        # Create single observation
        single_obs = np.zeros(self.single_num_obs, dtype=np.float32)
        single_obs[:3] = omega_scaled
        single_obs[3:6] = self.gravity_orientation
        single_obs[6:9] = self.cmd * self.cmd_scale
        single_obs[9: 9 + self.num_actions] = qj_scaled
        single_obs[9 + self.num_actions: 9 + 2 * self.num_actions] = dqj_scaled
        single_obs[9 + 2 * self.num_actions: 9 + 3 * self.num_actions] = self.action
        # single_obs[9 + 3 * self.num_actions: 9 + 3 * self.num_actions + 2] = np.array([sin_phase, cos_phase])
        return single_obs

    def policy_infer(self):
        single_obs = self.compute_observation()
        self.obs_history.append(single_obs)
        for i, hist_obs in enumerate(self.obs_history):
            start_idx = i * self.single_num_obs
            end_idx = start_idx + self.single_num_obs
            self.obs[start_idx:end_idx] = hist_obs

        obs_tensor = torch.from_numpy(self.obs).unsqueeze(0)

        self.action = self.policy(obs_tensor).detach().numpy().squeeze()
        self.action = np.clip(self.action, -1, 1)

        self.target_dof_pos = self.action * self.action_scale + self.default_angles
        target_dof_pos_pub = np.clip(self.target_dof_pos, -2.3, 2.3)

        tau = (target_dof_pos_pub - self.qj) * self.kp - self.dqj * self.kd

        for j in range(12):
            self.writer.add_scalars(
                f'fig/tau_{j}',  # 不包含 '/' 的 tag
                {
                    f'tau_pred_{j}': tau[j].item(),
                    f'tau_real_{j}': self.effort[j].item()
                },
                self.policy_count
            )
        self.control_data_pubisher_call(target_dof_pos_pub)

    def record(self):
        for j in range(6):
            target_dof_pos = self.target_dof_pos
            self.writer.add_scalars(
                f'fig_real/dof_pos_{j}_{j+6}',  # 不包含 '/' 的 tag
                {
                    f'cmd_{j}': target_dof_pos[j].item(),
                    f'cmd_{j+6}': target_dof_pos[j+6].item(),
                    f'state_{j}': controller.qj[j].item(),
                    f'state_{j+6}': controller.qj[j+6].item()
                },
                self.policy_count
            )
        
        for j in range(6):
            target_dof_pos = self.target_dof_pos
            self.writer.add_scalars(
                f'fig_real/dof_vel_{j}_{j+6}',  # 不包含 '/' 的 tag
                {
                    f'state_{j}': controller.dqj[j].item(),
                    f'state_{j+6}': controller.dqj[j+6].item()
                },
                self.policy_count
            )
        for j in range(3):
            self.writer.add_scalars(
                f'fig_real/imu_omega_mean',  # 不包含 '/' 的 tag
                {
                    f'{j}': np.mean(self.omega_history, axis=0)[j].item(),
                },
                self.policy_count)

        for j in range(3):
            self.writer.add_scalars(
                f'fig_real/imu_omega',  # 不包含 '/' 的 tag
                {
                    f'{j}': self.omega[j].item(),
                },
                self.policy_count)

        for j in range(4):
            self.writer.add_scalars(
                f'fig_real/imu_quat',  # 不包含 '/' 的 tag
                {
                    f'{j}': self.quat[j].item(),
                },
                self.policy_count)

        for j in range(3):
            self.writer.add_scalars(
                f'fig_real/gravity_orientation',  # 不包含 '/' 的 tag
                {
                    f'{j}': self.gravity_orientation[j].item(),
                },
                self.policy_count)


if __name__ == "__main__":
    # get config file name from command line
    import argparse

    controller = Controller()

    time0 = time.time()
    while True:
        if controller.is_over:
            break

        if controller.is_run:
            controller.policy_infer()

        else:
            init_pos = [0.0, 0.0, -0.1, -0.25, 0.17, 0.0,
                        0.0, 0.0, -0.1, -0.25, 0.17, 0.0]
            controller.control_data_pubisher_call(np.array(init_pos, dtype=np.float32))
        controller.policy_count += 1
        controller.control_counter += 1
        controller.record()

        time.sleep(controller.control_dt)

        if time.time() - time0 > 1:
            print(controller.control_counter, "control_hz")
            controller.control_counter = 0
            time0 = time.time()