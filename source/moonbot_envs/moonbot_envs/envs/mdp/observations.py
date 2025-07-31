from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from moonbot_envs.custom_lab_envs import CustomManagerBasedRLEnv
from moonbot_envs.envs.mdp.utils import get_base_height

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

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
#
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


def last_action(
    env: CustomManagerBasedRLEnv,
    module_no: int,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return the previous action for the given module."""

    group_name = "act_" + asset_cfg.name
    term_name = f"module_{module_no}_action"
    leg_term_name = term_name + "_leg"
    wheel_term_name = term_name + "_wheel"

    leg_act = env.action_manager.get_last_action(group_name, leg_term_name)
    wheel_act = env.action_manager.get_last_action(group_name, wheel_term_name)

    return torch.cat([leg_act, wheel_act], dim=-1)


# Integration Env Observation
def joint_vel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg, joint_ids: list[int]):
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_vel[:, joint_ids]


def joint_pos(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg, joint_ids: list[int]) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_pos[:, joint_ids]


from M2oE.configs import morphology_configs


def module_obs(env: CustomManagerBasedRLEnv, module_no: int, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    robot_name = asset_cfg.name

    joint_names = morphology_configs.joint_names_dict[robot_name][module_no]
    new_asset_cfg = SceneEntityCfg(name=robot_name)

    leg_joint_names = joint_names["leg"]
    wheel_joint_names = joint_names["wheel"]

    leg_ids = env.scene[robot_name].find_joints(leg_joint_names)[0]
    wheel_ids = env.scene[robot_name].find_joints(wheel_joint_names)[0]

    # observation terms
    _joint_pos = joint_pos(env, asset_cfg=new_asset_cfg, joint_ids=leg_ids + wheel_ids)
    _joint_vel = joint_vel(env, asset_cfg=new_asset_cfg, joint_ids=leg_ids + wheel_ids)
    _last_action = last_action(env, module_no=module_no, asset_cfg=new_asset_cfg)

    return torch.cat([
        _joint_pos,  # Select only the leg joints
        _joint_vel,
        _last_action,
    ], dim=-1)


def base_height_obs(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    base_height = torch.max(torch.tensor(0), get_base_height(env, asset_cfg))
    return base_height
