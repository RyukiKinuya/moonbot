from __future__ import annotations 

import numpy as np
import torch
from M2oE.configs import morphology_configs


def query_terrain_heights(xy_tensor, z_map, vertical_scale, terrain_cfg) -> torch.Tensor:
    # xy_tensor: (num_envs, 2)
    # z_map: (wildth, length)
    # return: (num_envs, 1)
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

def get_base_height(env, asset_cfg):
    """Base height observation."""
    asset = env.scene[asset_cfg.name]
    terrain = env.scene.terrain

    # get index of the robot from morphology_configs
    # z_map = terrain.terrain_generator.sub_terrain_heights[morphology_configs.morphology_list.index(asset_cfg.name)]
    # vertical_scale = terrain.cfg.terrain_generator.vertical_scale
    z_size = terrain.cfg.terrain_generator.size[2]

    asset_height_z = asset.data.root_pos_w[:, 2].unsqueeze(-1)

    z_base_offset = list(np.arange(-z_size, z_size * (len(morphology_configs.morphology_list) - 2) + 1e-6, z_size))

    base_height = asset_height_z - z_base_offset[morphology_configs.morphology_list.index(asset_cfg.name)]
    return base_height
