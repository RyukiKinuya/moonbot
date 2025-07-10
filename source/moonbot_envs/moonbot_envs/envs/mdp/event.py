from __future__ import annotations
from typing import TYPE_CHECKING

import torch
import numpy as np
import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObject, Articulation
from isaaclab.managers import SceneEntityCfg
from moonbot_envs.custom_lab_envs.terrains.terrain_importer import TerrainImporter
from M2oE.configs import morphology_configs

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

def reset_root_state_random(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
):
    for i, asset_name in enumerate(morphology_configs.morphology_list):
        asset: RigidObject = env.scene[asset_name]
        terrain = env.scene.terrain
        root_states = asset.data.default_root_state[env_ids].clone()

        terrain_size = terrain.cfg.terrain_generator.size[2]
        half_size = terrain_size / 2
        z_map = terrain.terrain_generator.sub_terrain_heights[i]
        
        vertical_scale = terrain.cfg.terrain_generator.vertical_scale

        z_base_offset = list(np.arange(-terrain_size, terrain_size * (len(morphology_configs.morphology_list) - 2) + 1e-6, terrain_size))


        env_origins = env.scene.env_origins[env_ids]
        
        xy_rand = torch.empty((len(env_ids), 2), device=asset.device)
        xy_rand.uniform_(-0.9*half_size, 0.9*half_size)

        rot_range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["roll", "pitch", "yaw"]]
        rot_ranges = torch.tensor(rot_range_list, device=asset.device)
        rot_rand_samples = math_utils.sample_uniform(
            rot_ranges[:, 0], rot_ranges[:, 1], (len(env_ids), 3), device=asset.device
        )

        positions_xy = env_origins[:, :2] + xy_rand

        z_query = query_terrain_heights(positions_xy, z_map, vertical_scale, terrain.cfg.terrain_generator)

        positions_z = root_states[:, 2:3] + z_query + z_base_offset[i % len(z_base_offset)]
        positions = torch.cat([positions_xy, positions_z], dim=1)

        orientations_delta = math_utils.quat_from_euler_xyz(
            rot_rand_samples[:, 0], rot_rand_samples[:, 1], rot_rand_samples[:, 2]
        )
        orientations = math_utils.quat_mul(root_states[:, 3:7], orientations_delta)

        range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=asset.device)
        rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)
        velocities = root_states[:, 7:13] + rand_samples

        # set into the physics simulation
        asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
        asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)


def query_terrain_heights(xy_tensor, z_map, vertical_scale, terrain_cfg) -> torch.Tensor:
    z_map = torch.from_numpy(z_map).to(xy_tensor.device)

    x_shift = terrain_cfg.size[0] / 2.0
    y_shift = terrain_cfg.size[1] / 2.0

    x_pix = ((xy_tensor[:, 0] + x_shift) / terrain_cfg.horizontal_scale)
    y_pix = ((xy_tensor[:, 1] + y_shift) / terrain_cfg.horizontal_scale)

    i = torch.clamp(x_pix.round().long(), 0, z_map.shape[0] - 1)
    j = torch.clamp(y_pix.round().long(), 0, z_map.shape[1] - 1)

    z_pixel = z_map[i, j]
    z = z_pixel * vertical_scale
    return z.unsqueeze(-1)  

