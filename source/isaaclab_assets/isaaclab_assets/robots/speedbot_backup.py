"""Configuration for Unitree robots.

The following configurations are available:

* :obj:`G1_CFG`: G1 humanoid robot
* :obj:`G1_MINIMAL_CFG`: G1 humanoid robot with minimal collision bodies
* :obj:`G1_29DOF_CFG`: G1 humanoid robot configured for locomanipulation tasks
* :obj:`G1_INSPIRE_FTP_CFG`: G1 29DOF humanoid robot with Inspire 5-finger hand


* :obj:`SBT_12DOF_CFG`: SBT humanoid robot configured for locomanipulation tasks

"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ActuatorNetMLPCfg, DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

#
"""Configuration for the Unitree G1 Humanoid robot."""


# G1_MINIMAL_CFG = G1_CFG.copy()
# G1_MINIMAL_CFG.spawn.usd_path = f"{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/G1/g1_minimal.usd"
"""Configuration for the Unitree G1 Humanoid robot with fewer collision meshes.

This configuration removes most collision meshes to speed up simulation.
"""


SBT_12DOF_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"source/isaaclab_assets/data/Robots/SPPEDrobots/speedbot_foot_box.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            fix_root_link=False,  # Configurable - can be set to True for fixed base
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.03),
        joint_pos={
            ".*_hip_pitch_joint": -0.10,
            ".*_knee_joint": -0.30,
            ".*_ankle_pitch_joint": 0.20,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": DCMotorCfg(
            joint_names_expr=[
                ".*_hip_roll_joint",
                ".*_hip_yaw_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
            ],
            effort_limit={
                ".*_hip_roll_joint": 267,
                ".*_hip_yaw_joint": 121,
                ".*_hip_pitch_joint": 267,
                ".*_knee_joint": 267,
            },
            velocity_limit={
                ".*_hip_roll_joint": 13.92,
                ".*_hip_yaw_joint": 7.17,
                ".*_hip_pitch_joint": 13.92,
                ".*_knee_joint": 13.92,
            },
            stiffness={
                ".*_hip_roll_joint": 150.0,
                ".*_hip_yaw_joint": 150.0,
                ".*_hip_pitch_joint": 150.0,
                ".*_knee_joint": 40.0,
            },
            damping={
                ".*_hip_roll_joint": 2.0,
                ".*_hip_yaw_joint": 2.0,
                ".*_hip_pitch_joint": 2.0,
                ".*_knee_joint": 4.0,
            },
            armature={
                ".*_hip_.*": 0.03,
                ".*_knee_joint": 0.03,
            },
            saturation_effort=18,
        ),
        "feet": DCMotorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            stiffness={
                ".*_ankle_pitch_joint": 20.0,
                ".*_ankle_roll_joint": 20.0,
            },
            damping={
                ".*_ankle_pitch_joint": 0.2, #2.0
                ".*_ankle_roll_joint": 0.1, #2.0
            },
            effort_limit={
                ".*_ankle_pitch_joint": 42.0,
                ".*_ankle_roll_joint": 42.0,
            },
            velocity_limit={
                ".*_ankle_pitch_joint": 6.7,
                ".*_ankle_roll_joint": 6.7,
            },
            armature=0.03,
            saturation_effort=80.0,
        ),
    },
    prim_path="/World/envs/env_.*/sbt_v1_3",
)
SBT_MINIMAL_CFG = SBT_12DOF_CFG.copy()
SBT_MINIMAL_CFG.spawn.usd_path = f"source/isaaclab_assets/data/Robots/SPPEDrobots/speedbot_foot_box.usd"
"""Configuration for the SBT Humanoid robot for locomanipulation tasks.

Usage examples:
    # For fixed base scenarios (upper body manipulation only)
    fixed_base_cfg = G1_29DOF_CFG.copy()
    fixed_base_cfg.spawn.articulation_props.fix_root_link = True

    # For mobile scenarios (locomotion + manipulation)
    mobile_cfg = G1_29DOF_CFG.copy()
    mobile_cfg.spawn.articulation_props.fix_root_link = False
"""
# SBT_CFG = ArticulationCfg(
#     spawn=sim_utils.UsdFileCfg(
#         usd_path=f"source/isaaclab_assets/data/Robots/SPPEDrobots/speedbot.usd",
#         activate_contact_sensors=True,
#         rigid_props=sim_utils.RigidBodyPropertiesCfg(
#             disable_gravity=False,
#             retain_accelerations=False,
#             linear_damping=0.0,
#             angular_damping=0.0,
#             max_linear_velocity=1000.0,
#             max_angular_velocity=1000.0,
#             max_depenetration_velocity=1.0,
#         ),
#         articulation_props=sim_utils.ArticulationRootPropertiesCfg(
#             enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=4
#         ),
#     ),
#     init_state=ArticulationCfg.InitialStateCfg(
#         pos=(0.0, 0.0, 0.75),
#         joint_pos={
#             ".*_hip_pitch_joint": -0.20,
#             ".*_knee_joint": 0.42,
#             ".*_ankle_pitch_joint": -0.23,
#             ".*_elbow_pitch_joint": 0.87,
#             "left_shoulder_roll_joint": 0.16,
#             "left_shoulder_pitch_joint": 0.35,
#             "right_shoulder_roll_joint": -0.16,
#             "right_shoulder_pitch_joint": 0.35,
#             "left_one_joint": 1.0,
#             "right_one_joint": -1.0,
#             "left_two_joint": 0.52,
#             "right_two_joint": -0.52,
#         },
#         joint_vel={".*": 0.0},
#     ),
#     soft_joint_pos_limit_factor=0.9,
#     actuators={
#         "legs": ImplicitActuatorCfg(
#             joint_names_expr=[
#                 ".*_hip_yaw_joint",
#                 ".*_hip_roll_joint",
#                 ".*_hip_pitch_joint",
#                 ".*_knee_joint",
#                 "torso_joint",
#             ],
#             effort_limit_sim=300,
#             stiffness={
#                 ".*_hip_yaw_joint": 150.0,
#                 ".*_hip_roll_joint": 150.0,
#                 ".*_hip_pitch_joint": 200.0,
#                 ".*_knee_joint": 200.0,
#                 "torso_joint": 200.0,
#             },
#             damping={
#                 ".*_hip_yaw_joint": 5.0,
#                 ".*_hip_roll_joint": 5.0,
#                 ".*_hip_pitch_joint": 5.0,
#                 ".*_knee_joint": 5.0,
#                 "torso_joint": 5.0,
#             },
#             armature={
#                 ".*_hip_.*": 0.01,
#                 ".*_knee_joint": 0.01,
#                 "torso_joint": 0.01,
#             },
#         ),
#         "feet": ImplicitActuatorCfg(
#             effort_limit_sim=20,
#             joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
#             stiffness=20.0,
#             damping=2.0,
#             armature=0.01,
#         ),
#         "arms": ImplicitActuatorCfg(
#             joint_names_expr=[
#                 ".*_shoulder_pitch_joint",
#                 ".*_shoulder_roll_joint",
#                 ".*_shoulder_yaw_joint",
#                 ".*_elbow_pitch_joint",
#                 ".*_elbow_roll_joint",
#                 ".*_five_joint",
#                 ".*_three_joint",
#                 ".*_six_joint",
#                 ".*_four_joint",
#                 ".*_zero_joint",
#                 ".*_one_joint",
#                 ".*_two_joint",
#             ],
#             effort_limit_sim=300,
#             stiffness=40.0,
#             damping=10.0,
#             armature={
#                 ".*_shoulder_.*": 0.01,
#                 ".*_elbow_.*": 0.01,
#                 ".*_five_joint": 0.001,
#                 ".*_three_joint": 0.001,
#                 ".*_six_joint": 0.001,
#                 ".*_four_joint": 0.001,
#                 ".*_zero_joint": 0.001,
#                 ".*_one_joint": 0.001,
#                 ".*_two_joint": 0.001,
#             },
#         ),
#     },
# )