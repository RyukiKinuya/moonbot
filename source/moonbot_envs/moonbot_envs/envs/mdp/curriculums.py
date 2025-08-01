# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import carb
from moonbot_envs.custom_lab_envs.custom_rl_env import CustomManagerBasedRLEnv
import omni.physics.tensors.impl.api as physx
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def terrain_levels_vel(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Curriculum based on the distance the robot walked when commanded to move at a desired velocity.

    This term is used to increase the difficulty of the terrain when the robot walks far enough and decrease the
    difficulty when the robot walks less than half of the distance required by the commanded velocity.

    .. note::
        It is only possible to use this term with the terrain type ``generator``. For further information
        on different terrain types, check the :class:`isaaclab.terrains.TerrainImporter` class.

    Returns:
        The mean terrain level for the given environment ids.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")
    # compute the distance the robot walked
    distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    # robots that walked far enough progress to harder terrains
    move_up = distance > terrain.cfg.terrain_generator.size[0] / 2
    # robots that walked less than half of their required distance go to simpler terrains
    move_down = distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    move_down *= ~move_up
    # update terrain levels
    terrain.update_env_origins(env_ids, move_up, move_down)
    # return the mean terrain level
    return torch.mean(terrain.terrain_levels.float())

def terrain_levels_dist(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    start_posi = asset.data.default_root_state[:3]
    posi_now = asset.data.root_pos_w[env_ids, :3] - env.scene.env_origins[env_ids, :3]
    distance = torch.norm(posi_now - start_posi, dim=1)
    full_dist = terrain.cfg.terrain_generator.size[1]

    move_up = distance > full_dist * 0.5
    move_down = distance < full_dist * 0.25
    move_down *= ~move_up

    terrain.update_env_origins(env_ids, move_up, move_down)
    return torch.mean(terrain.terrain_levels.float())

def modify_scene_gravity(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], gravity=(0.0, 0.0, -9.8100004196167), num_steps: int = 3000
):
    """Curriculum that modifies the gravity of the scene after a given number of steps.

    Args:
        env: The learning environment.
        env_ids: Not used since all environments are affected.
        gravity: The new gravity vector.
        num_steps: The number of steps after which the change should be applied.
    """
    if env.common_step_counter > num_steps:
        # set the gravity into the physics simulation
        physics_sim_view: physx.SimulationView = sim_utils.SimulationContext.instance().physics_sim_view
        physics_sim_view.set_gravity(carb.Float3(*gravity))

def modify_reward_weight_group(
    env: CustomManagerBasedRLEnv,
    env_ids: Sequence[int],
    group_name: str,
    term_name: str,
    weight: float,
    num_steps: int = 3000,
):
    if env.common_step_counter > num_steps:
        term_cfg = env.reward_manager.get_term_cfg(group_name, term_name)
        term_cfg.weight = weight
        env.reward_manager.set_term_cfg(group_name, term_name, term_cfg)
