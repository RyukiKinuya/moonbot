from __future__ import annotations

import torch
import math
from dataclasses import MISSING

import omni.isaac.contrib_tasks.moonbot.mdp as mdp

from isaaclab.utils import configclass

@configclass
class TriLeggedCommandsManipulateCfg:
    ee_pose = mdp.EePoseCommandCfg(
        asset_name="robot",
        ee_name="gripper_palm",

        resampling_time_range=(4.0, 4.0),
        debug_vis=True,
        ranges=mdp.EePoseCommandCfg.Ranges(
            dis_xy=(0.6, 1.0),
            pos_z=(1.2, 1.6),
            roll=(-3.14, 3.14),
            pitch=(-3.14, -3.14),
            yaw=(-3.14, -3.14),
        ),
    )

@configclass
class TriLeggedMovingTargetCommandCfg:
    ee_pose = mdp.EePoseMovingCommandCfg(
        asset_name="robot",
        ee_name = "gripper_palm",
        resampling_time_range=(8.0, 8.0),

        debug_vis=True,
        ranges=mdp.EePoseMovingCommandCfg.Ranges(
            start_point=(1.0, 0.0, 1.4),
            end_point=(3.0, 0.0, 1.4),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
    )

@configclass
class UniLeggedCommandsManipulateCfg:
    ee_pose = mdp.EePoseCommandCfg(
        asset_name="robot",
        ee_name="Arm_Link7",

        resampling_time_range=(4.0, 4.0),
        debug_vis=True,
        ranges=mdp.EePoseCommandCfg.Ranges(
            dis_xy=(0.5, 0.5),
            pos_z=(0.6, 1.2),
            roll=(-3.14, 3.14),
            pitch=(-3.14, 3.14),
            yaw=(-3.14, 3.14),
        ),
    )


@configclass
class UniLeggedUnlimitedCommandsCfg:
    ee_pose = mdp.EePoseCommandCfg(
        asset_name="robot",
        ee_name="Arm_Link7",

        resampling_time_range=(4.0, 4.0),
        debug_vis=True,
        ranges=mdp.EePoseCommandCfg.Ranges(
            dis_xy=(0.5, 1.2),
            pos_z=(0.0, 0.4),
            roll=(-3.14, 3.14),
            pitch=(-3.14, 3.14),
            yaw=(-3.14, 3.14),
        ),
    )


@configclass
class UniLeggedMovingTargetCommandCfg:
    ee_pose = mdp.EePoseMovingCommandCfg(
        asset_name="robot",
        ee_name = "Arm_Link7",
        resampling_time_range=(4.0, 4.0),

        debug_vis=True,
        ranges=mdp.EePoseMovingCommandCfg.Ranges(
            start_point=(1.0, 0.0, 1.4),
            end_point=(3.0, 0.0, 1.4),
            roll=(0.0, 0.0),
            pitch=(-1.57, -1.57),
            yaw=(0.0, 0.0),
        ),
    )

@configclass
class DragonLocomotionCommandCfg:
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(-0.3, 0.3), ang_vel_z=(-0.0, 0.0), heading=(-math.pi, math.pi)
        ),
    )

@configclass
class DragonReachingCommandCfg:
    # ee_pose = mdp.EePoseCommandCfg(
    #     asset_name="robot",
    #     ee_name="leg3gripper2_base",

    #     resampling_time_range=(4.0, 4.0),
    #     debug_vis=True,
    #     ranges=mdp.EePoseCommandCfg.Ranges(
    #         dis_xy=(0.2, 2.0),
    #         pos_z=(1.0, 1.6),
    #         roll=(-3.14, 3.14),
    #         pitch=(-3.14, 3.14),
    #         yaw=(-3.14, 3.14),
    #     ),
    # )
    ee_pose = mdp.EePoseMovingCommandCfg(
        asset_name="robot",
        ee_name = "leg3gripper2_base",
        resampling_time_range=(8.0, 8.0),

        debug_vis=True,
        ranges=mdp.EePoseMovingCommandCfg.Ranges(
            start_point=(1.0, 0.0, 1.0),
            end_point=(4.0, 0.0, 1.0),
            roll=(0.0, 0.0),
            pitch=(-1.57, -1.57),
            yaw=(3.14, 3.14),
        ),
    )