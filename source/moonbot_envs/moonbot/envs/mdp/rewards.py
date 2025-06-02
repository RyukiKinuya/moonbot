# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.assets import RigidObject
from isaaclab.assets import Articulation
from isaaclab.utils.math import quat_rotate_inverse, yaw_quat, matrix_from_quat, quat_error_magnitude, combine_frame_transforms, quat_error_magnitude, quat_mul
import numpy as np
import math
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_air_time_positive_biped(env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


# def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
#     # Penalize feet sliding
#     contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
#     contacts = contact_sensor.data.[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
#     asset = env.scene[asset_cfg.name]net_forces_w_history
#     body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
#     reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
#     return reward


def track_lin_vel_xy_yaw_frame_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_rotate_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)


def base_orient(
        env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    base_link_orient = asset.data.root_quat_w
    rot_m = matrix_from_quat(base_link_orient)
    z_axis_l = torch.matmul(rot_m, torch.tensor([0, 0, 1], dtype=torch.float, device=env.device)) #(num_env, 3)

    z_axis_w = torch.tensor([0, 0, 1], device=env.device).unsqueeze(0)    

    cos_theta = torch.sum(z_axis_l * z_axis_w, dim = 1)

    cos_theta_norm = (cos_theta + 1)/2

    reward = cos_theta_norm

    return reward
    

def base_height(
        env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), norm_factor=0.1
) -> torch.Tensor:
    if getattr(env.scene.sensors, "height_scanner", None) is not None:
        height_sensor = env.scene.sensors["height_scanner"]
        height = height_sensor.data.pos_w[:, 2].unsqueeze(1) - height_sensor.data.ray_hits_w[..., 2]
        height = height.mean(dim=1)
        target_height = env.command_manager.get_command("base_height")[:, 0]
        nheight = (height-target_height) / norm_factor
        reward = torch.exp(-nheight**2)
        return reward
    else:
        robot: RigidObject = env.scene[asset_cfg.name]
        base_height_command = env.command_manager.get_command("base_height")
        robot_pos = robot.data.root_pos_w[:, 2].unsqueeze(-1)

        dist_n = torch.norm(robot_pos - base_height_command, dim=-1)/norm_factor
        return torch.exp(-dist_n**2)
    

def base_vel_direction(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    base_vel = robot.data.root_lin_vel_w

    # get the direction of the base velocity
    base_vel_norm = torch.norm(base_vel, dim=-1)
    base_vel_dir = base_vel / base_vel_norm.unsqueeze(-1)

    # compare direction with the y-axis, the smaller the angle, the better
    y_axis = torch.tensor([0, 1, 0], device=env.device).unsqueeze(0)
    cos_theta = torch.sum(base_vel_dir * y_axis, dim=-1)
    cos_theta_norm = (cos_theta + 1)/2

    reward = cos_theta_norm

    return reward
        

def wheel_air_time(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    air_time_diff = torch.sum((last_air_time) * first_contact, dim=1)
    n_air_time_diff = air_time_diff / 0.5
    reward = torch.exp(-n_air_time_diff**2)
    return reward

def diff_from_init_pose(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    diffenece = robot.data.joint_pos - robot.data.default_joint_pos
    n_diff = torch.mean(diffenece, dim=-1) / 1.5708

    reward = torch.exp(-n_diff**2)

    return reward

def wheel_orient(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), 
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    
    wheel_body_idx = robot.find_bodies("leg.*_wheel_body")[0]
    wheel_body_rot = robot.data.body_state_w[:, wheel_body_idx, 3:7]
    command_orient = env.command_manager.get_command(command_name)[:, :2] #(lin_vel_x, lin_vel_y)
    command_orient = torch.cat([command_orient, torch.zeros_like(command_orient[:, 0]).unsqueeze(-1)], dim=-1).unsqueeze(1) #(num_env, 1, 3)
    command_orient = command_orient / torch.norm(command_orient, dim=-1, keepdim=True) #(num_env, 1, 3)

    rot_matrix = matrix_from_quat(wheel_body_rot)
    wheel_orient = torch.matmul(rot_matrix, torch.tensor([1, 0, 0], dtype=torch.float, device=env.device)) #feet orient (num_env, 3, 3)

    # difference between command_orient and feet_orien
    cos_theta = torch.sum(command_orient * wheel_orient, dim=-1) # (num_env, 3)
    cos_theta_norm = (cos_theta + 1)/2 # range [0, 1] (num_env, 3)

    reward = torch.mean(cos_theta_norm, dim=-1)
    return reward

def wheel_parallel(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    wheel_body_idx = robot.find_bodies("leg.*_wheel_body")[0]

    wheel_body_rot = robot.data.body_state_w[:, wheel_body_idx, 3:7]

    diff = quat_error_magnitude(wheel_body_rot[:, 0, :], wheel_body_rot[:, 1, :]) + \
           quat_error_magnitude(wheel_body_rot[:, 1, :], wheel_body_rot[:, 2, :]) + \
           quat_error_magnitude(wheel_body_rot[:, 2, :], wheel_body_rot[:, 0, :])
    # (0, 3pi)
    n_diff = diff / (math.pi * 3)

    reward = 1-n_diff

    return reward

def torque_rwd(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    torque = robot.data.applied_torque.norm(dim=-1)
    n_torque = torch.mean(torque, dim=-1) / 1000
    
    reward = torch.exp(-n_torque**2)

    return reward


# def wheel_command_velocity_reward(
#     env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), wheel_radius: float = 0.25
# ) -> torch.Tensor:
#     robot = env.scene[asset_cfg.name]
    
    
#     wheel_bodies = [
#         "leg2_wheel_left",
#         "leg2_wheel_right",
#         "leg1_wheel_left",
#         "leg3_wheel_left",
#         "leg1_wheel_right",
#         "leg3_wheel_right",
#     ]
    
#     command_vel = env.command_manager.get_command(command_name)[:, :2]
    
#     # 计算命令速度的模长
#     command_speed = torch.norm(command_vel, dim=1, keepdim=True)

#     total_reward = torch.zeros(env.num_envs, device=env.device)
    
#     for wheel in wheel_bodies:
        
#         ang_vel = robot.data.body_ang_vel_w[:, robot.find_bodies(wheel)[0], 2]  # 仅取z轴的角速度
        
#         # 计算轮子角速度乘以半径得到的线速度
#         wheel_lin_speed = ang_vel * wheel_radius
        
#         reward = torch.exp(-torch.abs(wheel_lin_speed - command_speed).sum(dim=1))
        
#         total_reward += reward
    
    
#     total_reward /= len(wheel_bodies)
    
#     return total_reward

def wheel_ang_velocity_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    
    wheel_bodies = [
        "leg2_wheel_left",
        "leg2_wheel_right",
        "leg1_wheel_left",
        "leg3_wheel_left",
        "leg1_wheel_right",
        "leg3_wheel_right",
    ]
    
    total_reward = torch.zeros(env.num_envs, device=env.device)
    
    for wheel in wheel_bodies:
        # from world frame to body frame
        body_ang_vel_b = quat_rotate_inverse(robot.data.body_state_w[:, robot.find_bodies(wheel)[0], 3:7].squeeze(1), robot.data.body_ang_vel_w[:, robot.find_bodies(wheel)[0], :].squeeze(1))

        reward_y = torch.exp(-torch.abs(body_ang_vel_b[:, 1]))
        penalty_z = torch.exp(-torch.abs(body_ang_vel_b[:, 2]))
        
        penalty_x = torch.exp(-torch.abs(body_ang_vel_b[:, 0]))
        
        total_reward += reward_y - 0.5*penalty_x - 0.5*penalty_z
    
    total_reward /= len(wheel_bodies)
    
    return total_reward


def wheel_joint_vel_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), wheel_name_expr: str = ".*wheel.*joint"
) -> torch.Tensor:
    # encourage the wheel joint to move
    robot = env.scene[asset_cfg.name]
    
    wheel_joint_idx = robot.find_joints(wheel_name_expr)[0]

    joint_vel = robot.data.joint_vel[:, wheel_joint_idx] # (num_env, num_wheel_joints)

    joint_vel_abs_mean = torch.mean(torch.abs(joint_vel), dim=-1)

    reward = 1 - torch.exp(-(joint_vel_abs_mean/3.14)**2)

    return reward


def tracking_goal_vel(
    env: ManagerBasedRLEnv, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]

    # 获取目标速度和机器人当前速度
    goal_vel = env.command_manager.get_command(command_name)[:, :2]
    robot_vel = robot.data.root_lin_vel_w[:, :2]

    target_vec_norm = torch.tensor([1.0, 0.0], device=robot_vel.device).unsqueeze(0)

    # 计算奖励值
    cur_vel = robot_vel
    reward = torch.minimum(torch.sum(target_vec_norm * cur_vel, dim=-1), goal_vel[:, 0]) / (goal_vel[:, 0] + 1e-5)

    return reward

def wheel_distances(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    

    wheel_body_idx = robot.find_bodies("leg.*_wheel_body")[0]
    base_idx = robot.find_bodies("base_link")[0][0]

    body_state_w = robot.data.body_state_w[:, wheel_body_idx, :3]  # (num_instances, num_wheel_bodies, 3)
    base_state_w = robot.data.body_state_w[:, base_idx, :3].squeeze(1)        # (num_instances, 3)

    dist_12 = torch.norm(body_state_w[:, 0] - body_state_w[:, 1], dim=-1)
    dist_13 = torch.norm(body_state_w[:, 0] - body_state_w[:, 2], dim=-1)
    dist_23 = torch.norm(body_state_w[:, 1] - body_state_w[:, 2], dim=-1)

    dist_base_1 = torch.norm(base_state_w - body_state_w[:, 0], dim=-1)
    dist_base_2 = torch.norm(base_state_w - body_state_w[:, 1], dim=-1)
    dist_base_3 = torch.norm(base_state_w - body_state_w[:, 2], dim=-1)


    mean_distance = (dist_12 + dist_13 + dist_23 + dist_base_1 + dist_base_2 + dist_base_3) / 6.0

    variance = (
        (dist_12 - mean_distance) ** 2 + 
        (dist_13 - mean_distance) ** 2 + 
        (dist_23 - mean_distance) ** 2 +
        (dist_base_1 - mean_distance) ** 2 + 
        (dist_base_2 - mean_distance) ** 2 + 
        (dist_base_3 - mean_distance) ** 2
    ) / 6.0

    reward = torch.exp(-variance**2)
    
    
    return reward.squeeze(-1)

def goal_distance_reward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
     robot = env.scene[asset_cfg.name]
     base_idx = robot.find_bodies("base_link")[0]

     body_state_w = robot.data.body_state_w[:, base_idx, :3]
     offset = torch.tensor([30, 0, 0.6], device='cuda').repeat(env.num_envs, 1)
     target_position = env.scene.env_origins + offset

     reward = 1.0 / (1.0 + torch.norm(body_state_w - target_position.unsqueeze_(1), dim=-1))

     return reward.squeeze(-1)

def wheel_stumble_penalty(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float =  1.0
) -> torch.Tensor:
    """Penalize feet for hitting vertical surfaces by analyzing net contact forces.

    This function penalizes the robot if the lateral (x, y) contact forces on the feet
    exceed a specified threshold relative to the vertical (z) forces. The purpose is to 
    discourage non-vertical contacts, which can indicate stumbling or unstable motion.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_forces_w = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    
    lateral_forces = torch.norm(net_forces_w[:, :, :2], dim=2)
    vertical_forces = torch.abs(net_forces_w[:, :, 2])
    
    # Penalize cases where lateral forces exceed the threshold relative to vertical forces
    penalty_condition = lateral_forces > threshold * vertical_forces

    reward = torch.any(penalty_condition, dim=1).float()
    
    return reward


def uni_position_command_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg, base_name:str) -> torch.Tensor:
    """Penalize tracking of the position error using L2-norm.

    The function computes the position error between the desired position (from the command) and the
    current position of the asset's body (in world frame). The position error is computed as the L2-norm
    of the difference between the desired and current positions.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions
    des_pos_w = command[:, :3] + env.scene.env_origins
    curr_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3]  # type: ignore
    return torch.norm(curr_pos_w - des_pos_w, dim=1)


def uni_position_command_error_tanh(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg, base_name:str
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions
    des_pos_w = command[:, :3] + env.scene.env_origins
    curr_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3]  # type: ignore
    distance = torch.norm(curr_pos_w - des_pos_w, dim=1)
    return 1 - torch.tanh(distance / std)


def uni_orientation_command_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg, base_name:str
                              ) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_quat_w = command[:, 3:7]
    curr_quat_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], 3:7]  # type: ignore
    return quat_error_magnitude(curr_quat_w, des_quat_w)

def ee_velocity(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), norm_factor=5
) -> torch.Tensor:
    robot: RigidObject = env.scene[asset_cfg.name]
    ee_link_idx = robot.find_bodies("gripper_palm")[0][0]

    ee_velo = robot.data.body_lin_vel_w[:, ee_link_idx]
    ee_velo_n = torch.norm(ee_velo, dim=-1)/norm_factor

    return torch.exp(-ee_velo_n**2)
    
def base_dist(env:ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), norm_factor=2) -> torch.Tensor:
    # minimize the xy-plane dist
    robot: RigidObject = env.scene[asset_cfg.name]
    ee_pose_command = env.command_manager.get_command("ee_pose")[:, :2]
    robot_pos = robot.data.root_pos_w[:, :2]

    dist_n = torch.norm(robot_pos - ee_pose_command, dim=-1)/norm_factor

    return torch.exp(-dist_n**2)

# def joint_vel_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
#     """Penalize joint velocities on the articulation using L1-kernel.

#     NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint velocities contribute to the L1 norm.
#     """
#     # extract the used quantities (to enable type-hinting)
#     asset: Articulation = env.scene[asset_cfg.name]
#     return torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)

def position_command_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize tracking of the position error using L2-norm.

    The function computes the position error between the desired position (from the command) and the
    current position of the asset's body (in world frame). The position error is computed as the L2-norm
    of the difference between the desired and current positions.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_state_w[:, :3], asset.data.root_state_w[:, 3:7], des_pos_b)
    curr_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3]  # type: ignore
    return torch.norm(curr_pos_w - des_pos_w, dim=1)


def position_command_error_tanh(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward tracking of the position using the tanh kernel.

    The function computes the position error between the desired position (from the command) and the
    current position of the asset's body (in world frame) and maps it with a tanh kernel.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_state_w[:, :3], asset.data.root_state_w[:, 3:7], des_pos_b)
    curr_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3]  # type: ignore
    distance = torch.norm(curr_pos_w - des_pos_w, dim=1)
    return 1 - torch.tanh(distance / std)


def orientation_command_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize tracking orientation error using shortest path.

    The function computes the orientation error between the desired orientation (from the command) and the
    current orientation of the asset's body (in world frame). The orientation error is computed as the shortest
    path between the desired and current orientations.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current orientations
    des_quat_b = command[:, 3:7]
    des_quat_w = quat_mul(asset.data.root_state_w[:, 3:7], des_quat_b)
    curr_quat_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], 3:7]  # type: ignore
    return quat_error_magnitude(curr_quat_w, des_quat_w)


def uni_leg_flat_orientation_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize non-flat base orientation using L2-kernel.

    This is computed by penalizing the xy-components of the projected gravity vector.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    base_link_posi = asset.data.root_pos_w
    arm_link_1_posi = asset.data.body_state_w[:, asset.find_bodies("Arm_Link1")[0][0], :3]

    vec = arm_link_1_posi - base_link_posi

    return torch.sum(torch.square(vec[:, :2]), dim=1)

def base_dist_from_origin(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    base_link_posi = asset.data.root_pos_w
    env_origin = env.scene.env_origins

    return torch.norm(base_link_posi - env_origin, dim=1)

def penalize_low_velocity(
    env: ManagerBasedRLEnv, threshold: float = 0.2, penalty_scale: float = 1.0, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    lin_vel_xy = torch.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    penalty = torch.where(lin_vel_xy < threshold, -penalty_scale * (threshold - lin_vel_xy), torch.zeros_like(lin_vel_xy))
    return penalty

def dragon_flat_orientation_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize non-flat base orientation using L2-kernel.

    This is computed by penalizing the xy-components of the projected gravity vector.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    base_link_posi = asset.data.root_pos_w
    arm_link_1_posi = asset.data.body_state_w[:, asset.find_bodies("leg4link1")[0][0], :3]


    base_link2_posi = asset.data.body_state_w[:, asset.find_bodies("leg4link6")[0][0], :3]
    arm_link_2_posi = asset.data.body_state_w[:, asset.find_bodies("wheel14_body")[0][0], :3]

    vec1 = arm_link_1_posi - base_link_posi
    vec2 = arm_link_2_posi - base_link2_posi

    # return torch.sum(torch.square(vec[:, :2]), dim=1)
    return torch.sum(torch.square(vec1[:, :2]) + torch.square(vec2[:, :2]), dim=1)


def wheel_same_act(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]

    wheel1_vel = asset.data.joint_vel[:, asset.find_joints("wheel12.*joint")[0]]
    wheel2_vel = asset.data.joint_vel[:, asset.find_joints("wheel14.*joint")[0]]
    
    error1 = torch.norm(wheel1_vel[:, 0] - wheel1_vel[:, 1])
    error2 = torch.norm(wheel2_vel[:, 0] - wheel2_vel[:, 1])

    return (error1 + error2) / 10
    
def wheel_on_ground(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold) -> torch.Tensor:
    """Penalize undesired contacts as the number of violations that are above a threshold."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # check if contact force is above threshold
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    # sum over contacts for each environment
    return torch.sum(is_contact, dim=1)