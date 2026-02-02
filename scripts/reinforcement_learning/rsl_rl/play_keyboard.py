# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to control robot with keyboard in Isaac Lab visualization."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Control robot with keyboard in Isaac Lab visualization.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

# 导入键盘控制器
from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Control robot with keyboard."""
    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # Print DOF order information
    if hasattr(env.unwrapped, "scene") and hasattr(env.unwrapped.scene, "articulations"):
        for articulation_name, articulation in env.unwrapped.scene.articulations.items():
            if hasattr(articulation, "data") and hasattr(articulation.data, "joint_names"):
                print(f"[DOF Order] {articulation_name}: {articulation.data.joint_names}")
    
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join("logs", "keyboard_control", "videos"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during keyboard control.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap environment
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # 初始化键盘控制器
    print("[INFO]: Initializing keyboard controller...")
    keyboard_cfg = Se2KeyboardCfg(
        v_x_sensitivity=0.8,    # x轴线性速度灵敏度
        v_y_sensitivity=0.4,    # y轴线性速度灵敏度
        omega_z_sensitivity=1.0  # z轴角速度灵敏度
    )
    keyboard = Se2Keyboard(keyboard_cfg)
    
    print("\nKeyboard Control Instructions:")
    print("============================")
    print("Forward/Backward: Numpad 8 / 2 or Up/Down Arrow")
    print("Left/Right: Numpad 4 / 6 or Left/Right Arrow")
    print("Rotate: Numpad 7 / 9 or Z / X")
    print("Reset: L")
    print("============================\n")

    dt = env.unwrapped.step_dt

    # reset environment
    env.reset()
    keyboard.reset()
    timestep = 0
    
    # 获取动作空间信息
    action_space = env.action_space
    print(f"[INFO]: Action space: {action_space}")
    
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # 获取键盘命令
            keyboard_command = keyboard.advance()
            
            # 将键盘命令转换为环境所需的动作格式
            # 对于移动机器人环境，命令通常是 [vx, vy, wz]
            actions = keyboard_command.repeat(env.num_envs, 1)
            
            # 确保动作在动作空间范围内
            actions = torch.clamp(actions, action_space.low, action_space.high)
            
            # 执行环境步骤
            obs, _, dones, _ = env.step(actions)
            
            # 处理环境重置（如果需要）
            if dones.any():
                env.reset()
        
        # 实时控制的时间延迟
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)
        
        if args_cli.video:
            timestep += 1
            # 录制一个视频后退出
            if timestep == args_cli.video_length:
                break

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()