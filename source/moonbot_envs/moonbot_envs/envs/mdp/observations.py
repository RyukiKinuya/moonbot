from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster

from isaaclab.envs import mdp
from moonbot_envs.custom_lab_envs import CustomManagerBasedRLEnv



if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv



def body_ang_vel(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Observes the angular velocity of the wheels in the robot's root frame."""

    # extract the used quantities (to enable type-hinting)
    robot = env.scene["robot"]
    wheel_bodies = [
        "leg2_wheel_left",
        "leg2_wheel_right",
        "leg1_wheel_left",
        "leg3_wheel_left",
        "leg1_wheel_right",
        "leg3_wheel_right",
    ]

    # Initialize a list to store angular velocities
    ang_velocities = []

    for wheel in wheel_bodies:
        # 获取轮子的角速度 (z轴分量)
        ang_vel = robot.data.body_ang_vel_w[:, robot.find_bodies(wheel)[0], 2]
        ang_velocities.append(ang_vel)

    # 堆叠
    ang_vel_tensor = torch.cat(ang_velocities, dim=1)

    return ang_vel_tensor

# def wheel_parallel(env: ManagerBasedRLEnv) -> torch.Tensor:
#     """Observes the parallelism of the wheels in the robot's root frame."""

#     # extract the used quantities (to enable type-hinting)
#     robot = env.scene["robot"]
#     wheel_body_idx = robot.find_bodies("leg.*_wheel_body")  # 假设返回三个索引

#     # Initialize a list to store the parallelism data
#     parallelism = []

#     for idx in wheel_body_idx:
#         # 获取轮子的旋转数据 (四元数，有 4 个分量)
#         wheel_body_rot = robot.data.body_state_w[:, idx, 3:7]

#         # 提取旋转数据的每个分量，展平并加入列表
#         parallelism.append(wheel_body_rot.view(-1))  # 将 4 个分量展平

#     # 将所有旋转数据的列表转换为张量，并展平为一维张量
#     parallelism_tensor = torch.cat(parallelism).view(-1)

#     return parallelism_tensor

def ee_pose(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), ee_name="gripper_palm"):
    asset: RigidObject = env.scene[asset_cfg.name]
    # obtain the desired and current positions
    curr_pos_w = asset.data.body_state_w[:, asset.find_bodies(ee_name)[0][0], :3]
    curr_posi_b = curr_pos_w[:, :3] - asset.data.root_pos_w[:, :3]
    curr_quat_b = curr_pos_w[:, 3:7]


    return torch.cat([curr_posi_b, curr_quat_b], dim=-1)

def base_height(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    asset: Articulation = env.scene[asset_cfg.name]
    obs = asset.data.root_pos_w[:, 2].unsqueeze(-1)
    return obs

def ee_pose_command(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """The generated command from command term in the command manager with the given name."""
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_posi_w = command[:, :3] + env.scene.env_origins
    des_posi_b = des_posi_w - asset.data.root_pos_w[:, :3]
    des_quat_b = command[:, 3:7]

    return torch.cat([des_posi_b, des_quat_b], dim=-1)


def _last_action(env: CustomManagerBasedRLEnv, group_name: str | None = None, action_name: str | None = None):
    if group_name is None:
        return env.action_manager.action
    else:
        if action_name is None:
            return env.action_manager.get_group_action(group_name)
        else:
            return env.action_manager.get_term(group_name, action_name).action

# Integration Env Observation
from M2oE.configs import morphology_configs
def module_obs(env: ManagerBasedEnv, module_no:int, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    robot_name = asset_cfg.name

    joint_expr = morphology_configs.joint_expr_dict[robot_name][module_no]

    # observation terms
    joint_pos = mdp.joint_pos(env, asset_cfg=asset_cfg.replace(joint_names=[joint_expr["leg"], joint_expr["wheel"]]))
    joint_vel = mdp.joint_vel(env, asset_cfg=asset_cfg.replace(joint_names=[joint_expr["leg"], joint_expr["wheel"]]))

    return torch.cat([
        joint_pos,
        joint_vel], dim=-1)