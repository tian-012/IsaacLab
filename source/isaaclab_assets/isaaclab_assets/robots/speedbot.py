import isaaclab.sim as sim_utils
from isaaclab.actuators import ActuatorNetMLPCfg, DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

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
            ".*_knee_joint": -0.25,
            ".*_ankle_pitch_joint": 0.17,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_roll_joint",
                ".*_hip_yaw_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
            ],
            effort_limit_sim=300.0,
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
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            effort_limit_sim=80.0,
            stiffness={
                ".*_ankle_pitch_joint": 20.0,
                ".*_ankle_roll_joint": 20.0,
            },
            damping={
                ".*_ankle_pitch_joint": 2.0,
                ".*_ankle_roll_joint": 2.0,
            },
            armature=0.03,
        ),
    },
    prim_path="/World/envs/env_.*/sbt_v1_3",
)
SBT_MINIMAL_CFG = SBT_12DOF_CFG.copy()
SBT_MINIMAL_CFG.spawn.usd_path = f"source/isaaclab_assets/data/Robots/SPPEDrobots/speedbot_foot_box.usd"
