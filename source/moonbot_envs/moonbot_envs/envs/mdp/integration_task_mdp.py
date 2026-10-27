"""Task observations and rewards for multi-morphology MoonBot integration tasks."""
from __future__ import annotations

import torch
from isaaclab.managers import SceneEntityCfg
from M2oE.configs import morphology_configs
import moonbot_envs.envs.mdp as mdp


OBSTACLE_CENTERS = ((0.65, -0.25, 0.95), (0.90, 0.0, 1.05), (0.65, 0.25, 0.95))


def module_task_obs(env, asset_cfg: SceneEntityCfg, module_no: int, command_name: str,
                    end_effector_name: str, with_obstacles: bool = False):
    """Per-module proprioception plus goal-relative EE state and optional obstacle state."""
    robot = env.scene[asset_cfg.name]
    base_obs = mdp.module_obs(env, module_no=module_no, asset_cfg=asset_cfg)
    ee_idx = robot.find_bodies(end_effector_name)[0][0]
    ee_pos = robot.data.body_pos_w[:, ee_idx] - env.scene.env_origins
    goal_pos = env.command_manager.get_command(command_name)[:, :3]
    goal_rel_ee = goal_pos - ee_pos
    goal_rel_base = goal_pos - (robot.data.root_pos_w[:, :3] - env.scene.env_origins)
    chunks = [base_obs, goal_rel_base, goal_rel_ee]
    if with_obstacles:
        centers = torch.tensor(OBSTACLE_CENTERS, device=env.device, dtype=ee_pos.dtype)
        chunks.append((centers.unsqueeze(0) - ee_pos.unsqueeze(1)).reshape(env.num_envs, -1))
    return torch.cat(chunks, dim=-1)


def reach_distance(env, asset_cfg: SceneEntityCfg, command_name: str, end_effector_name: str):
    robot = env.scene[asset_cfg.name]
    ee_idx = robot.find_bodies(end_effector_name)[0][0]
    ee_pos = robot.data.body_pos_w[:, ee_idx] - env.scene.env_origins
    goal_pos = env.command_manager.get_command(command_name)[:, :3]
    return torch.linalg.vector_norm(goal_pos - ee_pos, dim=-1)


def reach_progress(env, asset_cfg: SceneEntityCfg, command_name: str, end_effector_name: str,
                   std: float = 0.15):
    # The exponential shaping adapts Alessi et al. (2024)'s EE-to-target distance reward.
    return torch.exp(-reach_distance(env, asset_cfg, command_name, end_effector_name) / std)


def reach_success(env, asset_cfg: SceneEntityCfg, command_name: str, end_effector_name: str,
                  tolerance: float = 0.06):
    return (reach_distance(env, asset_cfg, command_name, end_effector_name) < tolerance).float()


def all_morphologies_reached(env, tolerance: float = 0.06):
    """End the shared episode only when each morphology has reached its own goal."""
    reached = []
    for asset_name in morphology_configs.morphology_list:
        ee_name = morphology_configs.ee_body_name_dict[asset_name]
        # Tricycle and Dragon use the gripper palm; Minimal's configured EE differs.
        if asset_name == "moonbot_minimal":
            ee_name = "Arm_Link7"
        reached.append(reach_distance(env, SceneEntityCfg(asset_name), f"ee_pose_{asset_name}", ee_name) < tolerance)
    return torch.stack(reached, dim=0).all(dim=0)


def obstacle_clearance(env, asset_cfg: SceneEntityCfg, command_name: str, end_effector_name: str,
                       margin: float = 0.18):
    # Tang et al. (2022) use obstacle-position state and collision/progress terms in navigation;
    # this is a continuous safety-margin adaptation for the MoonBot manipulator task.
    robot = env.scene[asset_cfg.name]
    ee_idx = robot.find_bodies(end_effector_name)[0][0]
    ee_pos = robot.data.body_pos_w[:, ee_idx] - env.scene.env_origins
    centers = torch.tensor(OBSTACLE_CENTERS, device=env.device, dtype=ee_pos.dtype)
    distances = torch.linalg.vector_norm(centers.unsqueeze(0) - ee_pos.unsqueeze(1), dim=-1)
    return torch.square(torch.clamp(margin - distances, min=0.0)).sum(dim=-1)


def obstacle_collision(env, sensor_cfg: SceneEntityCfg, threshold: float = 12.0):
    sensor = env.scene.sensors[sensor_cfg.name]
    force = sensor.data.net_forces_w
    return (torch.linalg.vector_norm(force, dim=-1).amax(dim=-1) > threshold).float()
