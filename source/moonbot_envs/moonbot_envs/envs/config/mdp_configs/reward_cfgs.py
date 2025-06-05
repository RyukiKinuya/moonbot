from __future__ import annotations

from dataclasses import MISSING

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
import moonbot_envs.moonbot_envs.mdp as mdp
import math

from isaaclab.utils import configclass

@configclass
class RewardsManipulateStandingCfg:
    """Reward terms for the MDP."""
    # task terms
    end_effector_position_tracking = RewTerm(
        func=mdp.uni_position_command_error,
        weight=-1,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="gripper_palm"), "command_name": "ee_pose", "base_name": "base_link"},
    )
    end_effecstor_position_tracking_fine_grained = RewTerm(
        func=mdp.uni_position_command_error_tanh,
        weight=0.5,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="gripper_palm"), "std": 0.1, "command_name": "ee_pose", "base_name": "base_link"},
    )
    end_effector_orientation_tracking = RewTerm(
        func=mdp.uni_orientation_command_error,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="gripper_palm"), "command_name": "ee_pose", "base_name": "base_link"},
    )

    # action penalty
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.005)

    is_alive = RewTerm(
        func=mdp.is_alive,
        weight=5.0,
    )

    # base_spd_direction = RewTerm(
    #     func=mdp.base_vel_direction,
    #     weight=2.0,
    #     params={"asset_cfg": SceneEntityCfg("robot", body_names="base_link")},
    # )

    # joint_acc = RewTerm(func = mdp.joint_acc_l2, weight = -0.0001, params = {"asset_cfg": SceneEntityCfg("robot")})

    # joint_torque = RewTerm(func = mdp.joint_torques_l2, weight = -0.00001, params = {"asset_cfg": SceneEntityCfg("robot")})

    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.01,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names="(?!.*wheel.*).*joint.*")},
    )

    # diff_from_init_pose = RewTerm(
    #     func=mdp.diff_from_init_pose,
    #     params={"asset_cfg": SceneEntityCfg("robot", body_names="(?!.*wheel.*)leg.*")},
    #     weight=1.5
    # )

    # base_velocities = RewTerm(
    #     func=mdp.penalize_low_velocity,
    #     weight=10.0,
    #     params={"asset_cfg": SceneEntityCfg("robot", body_names="base_link")},
    # )

    illegal_contact = RewTerm(
        func=mdp.illegal_contact,
        weight=-10.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="(?!.*wheel.*).*link.*"), "threshold": 8.0},
    )

    # project_gravity = RewTerm(func=mdp.flat_orientation_l2, weight=-10.0)

    # wheel_velocity_rwd = RewTerm(
    #     func=mdp.wheel_joint_vel_reward,
    #     weight=0.05,

    # )

    # wheel_parrallel = RewTerm(func=mdp.wheel_parallel, weight=0.5)


@configclass
class RewardsManipulateCfg:
    """Reward terms for the MDP."""
    # task terms
    end_effector_position_tracking = RewTerm(
        func=mdp.uni_position_command_error,
        weight=-10,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="gripper_palm"), "command_name": "ee_pose", "base_name": "base_link"},
    )
    end_effecstor_position_tracking_fine_grained = RewTerm(
        func=mdp.uni_position_command_error_tanh,
        weight=5.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="gripper_palm"), "std": 0.1, "command_name": "ee_pose", "base_name": "base_link"},
    )
    end_effector_orientation_tracking = RewTerm(
        func=mdp.uni_orientation_command_error,
        weight=-5.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="gripper_palm"), "command_name": "ee_pose", "base_name": "base_link"},
    )

    # action penalty
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.0001)

    is_alive = RewTerm(
        func=mdp.is_alive,
        weight=20.0,
    )

    # base_spd_direction = RewTerm(
    #     func=mdp.base_vel_direction,
    #     weight=2.0,
    #     params={"asset_cfg": SceneEntityCfg("robot", body_names="base_link")},
    # )

    joint_acc = RewTerm(func = mdp.joint_acc_l2, weight = -0.00001, params = {"asset_cfg": SceneEntityCfg("robot")})

    # joint_torque = RewTerm(func = mdp.joint_torques_l2, weight = -0.0001, params = {"asset_cfg": SceneEntityCfg("robot")})

    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.0001,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names="(?!.*wheel.*).*joint.*")},
    )

    diff_from_init_pose = RewTerm(
        func=mdp.diff_from_init_pose,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="(?!.*wheel.*)leg.*")},
        weight=5.0
    )

    # base_velocities = RewTerm(
    #     func=mdp.penalize_low_velocity,
    #     weight=10.0,
    #     params={"asset_cfg": SceneEntityCfg("robot", body_names="base_link")},
    # )

    illegal_contact = RewTerm(
        func=mdp.illegal_contact,
        weight=-10.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="(?!.*wheel.*).*link.*"), "threshold": 8.0},
    )

    project_gravity = RewTerm(func=mdp.flat_orientation_l2, weight=-10.0)

    wheel_velocity_rwd = RewTerm(
        func=mdp.wheel_joint_vel_reward,
        weight=0.1,
    )

    # wheel_parrallel = RewTerm(func=mdp.wheel_parallel, weight=0.5)


@configclass
class UniLeggedManipulateRewardsCfg:
    """Reward terms for the MDP."""

    # task terms
    end_effector_position_tracking = RewTerm(
        func=mdp.uni_position_command_error,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="Arm_Link7"), "command_name": "ee_pose", "base_name": "base_link"},
    )
    end_effecstor_position_tracking_fine_grained = RewTerm(
        func=mdp.uni_position_command_error_tanh,
        weight=1.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="Arm_Link7"), "std": 0.1, "command_name": "ee_pose", "base_name": "base_link"},
    )
    end_effector_orientation_tracking = RewTerm(
        func=mdp.uni_orientation_command_error,
        weight=-0.25,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="Arm_Link7"), "command_name": "ee_pose", "base_name": "base_link"},
    )

    # action penalty
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.001)

    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.001,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    illegal_contact = RewTerm(
        func=mdp.illegal_contact,
        weight=-10.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*"), "threshold": 8.0},
    )

    is_alive = RewTerm(
        func=mdp.is_alive,
        weight=5.0,
    )


@configclass
class UniLeggedStandingReachingRewardsCfg:
    # stage 1
    # diff_from_init_pose = RewTerm(
    #     func=mdp.diff_from_init_pose,
    #     params={"asset_cfg": SceneEntityCfg("robot")},
    #     weight=10.0
    # )

    base_balance = RewTerm(
        func=mdp.uni_leg_flat_orientation_l2,
        weight=-5.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="base_link")},
    )

    # stage 2
    # base_dis_from_origin = RewTerm(
    #     func=mdp.base_dist_from_origin,
    #     weight=-0.0,
    #     params={"asset_cfg": SceneEntityCfg("robot", body_names="base_link")},
    # )

    # base_vel = RewTerm(
    #     func=mdp.body_lin_acc_l2,
    #     weight=-0.0,
    #     params={"asset_cfg": SceneEntityCfg("robot", body_names="base_link")},
    # )
    
    is_alive = RewTerm(
        func=mdp.is_alive,
        weight=0.0,
    )

    # stage 3
    end_effector_position_tracking = RewTerm(
        func=mdp.uni_position_command_error,
        weight=-0.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="Arm_Link7"), "command_name": "ee_pose", "base_name": "base_link"},
    )
    end_effecstor_position_tracking_fine_grained = RewTerm(
        func=mdp.uni_position_command_error_tanh,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="Arm_Link7"), "std": 0.1, "command_name": "ee_pose", "base_name": "base_link"},
    )
    end_effector_orientation_tracking = RewTerm(
        func=mdp.uni_orientation_command_error,
        weight=-0.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="Arm_Link7"), "command_name": "ee_pose", "base_name": "base_link"},
    )

    # action penalty
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.001)

    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.0001,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names="joint.*")},
    )

    illegal_contact = RewTerm(
        func=mdp.illegal_contact,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="(?!Wheel.*).*"), "threshold": 8.0},
    )


@configclass
class UniLeggedUnlimitedReachingRewardsCfg:
    joint_vel_limit = RewTerm(
        func=mdp.joint_vel_limits,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*(joint).*"), "soft_ratio": 0.9},
    )

    # base_balance = RewTerm(
    #     func=mdp.uni_leg_flat_orientation_l2,
    #     weight=-2.0,
    #     params={"asset_cfg": SceneEntityCfg("robot", body_names="base_link")},
    # )

    end_effector_position_tracking = RewTerm(
        func=mdp.uni_position_command_error,
        weight=-5.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="Arm_Link7"), "command_name": "ee_pose", "base_name": "base_link"},
    )
    end_effecstor_position_tracking_fine_grained = RewTerm(
        func=mdp.uni_position_command_error_tanh,
        weight=2.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="Arm_Link7"), "std": 0.1, "command_name": "ee_pose", "base_name": "base_link"},
    )
    end_effector_orientation_tracking = RewTerm(
        func=mdp.uni_orientation_command_error,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="Arm_Link7"), "command_name": "ee_pose", "base_name": "base_link"},
    )

    is_alive = RewTerm(
        func=mdp.is_alive,
        weight=10.0,
    )

    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.0001)

    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.0001,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    joint_vel_limit = RewTerm(
        func=mdp.joint_vel_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*(joint).*"), "soft_ratio": 0.9},
    )

    illegal_contact = RewTerm(
        func=mdp.illegal_contact,
        weight=-0.5,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="(?!Wheel.*).*"), "threshold": 1.0},
    )

@configclass
class DragonLocomotionRewardsCfg:
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=10.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )

    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=5.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )

    # lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    # ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.1)

    is_alive = RewTerm(func=mdp.is_alive, weight=1.0)

    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.5,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*leg.*"), "threshold": 1.0},
    )

    wheel_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=0.5,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="wheel12_left|wheel12_right|wheel14_left|wheel14_right"), "threshold": 1.0},
    )
    # dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=0.0)

    diff_from_init_pose = RewTerm(
        func=mdp.diff_from_init_pose,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*leg.*")},
        weight=1.0
    )

    base_balance = RewTerm(
        func=mdp.dragon_flat_orientation_l2,
        weight=-0.1,
    )

    wheel_velocity_rwd = RewTerm(
        func=mdp.wheel_joint_vel_reward,
        weight=2.0,
        params={"wheel_name_expr": "wheel.*joint"},
    )


@configclass
class DragonReachingRewardsCfg:
    # task terms
    end_effector_position_tracking = RewTerm(
        func=mdp.uni_position_command_error,
        weight=-10,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="leg3gripper2_base"), "command_name": "ee_pose", "base_name": "base_link"},
    )
    end_effecstor_position_tracking_fine_grained = RewTerm(
        func=mdp.uni_position_command_error_tanh,
        weight=5,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="leg3gripper2_base"), "std": 0.1, "command_name": "ee_pose", "base_name": "base_link"},
    )
    end_effector_orientation_tracking = RewTerm(
        func=mdp.uni_orientation_command_error,
        weight=-5,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="leg3gripper2_base"), "command_name": "ee_pose", "base_name": "base_link"},
    )

    # action penalty
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.0001)

    is_alive = RewTerm(
        func=mdp.is_alive,
        weight=20.0,
    )

    # wheel_act = RewTerm(func=mdp.wheel_same_act, weight=-0.01)
    wheel_vel = RewTerm(func=mdp.wheel_joint_vel_reward, weight=0.001)
    wheel_on_ground = RewTerm(func=mdp.wheel_on_ground, 
                              params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="wheel(12|14)_(left|right)"), "threshold": 8.0},
                              weight=1.0)

    # base_spd_direction = RewTerm(
    #     func=mdp.base_vel_direction,
    #     weight=2.0,
    #     params={"asset_cfg": SceneEntityCfg("robot", body_names="base_link")},
    # )

    # joint_acc = RewTerm(func = mdp.joint_acc_l2, weight = -0.00001, params = {"asset_cfg": SceneEntityCfg("robot")})
    # dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-4)
    # dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-6)
    # action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    # joint_torque = RewTerm(func = mdp.joint_torques_l2, weight = -0.0001, params = {"asset_cfg": SceneEntityCfg("robot")})

    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.0001,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*leg.*")},
    )

    diff_from_init_pose = RewTerm(
        func=mdp.diff_from_init_pose,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*leg.*")},
        weight=0.1
    )

    base_balance = RewTerm(
        func=mdp.dragon_flat_orientation_l2,
        weight=-0.2,
    )

    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.01,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="(?!.*wheel.*).*link.*"), "threshold": 8.0},
    )