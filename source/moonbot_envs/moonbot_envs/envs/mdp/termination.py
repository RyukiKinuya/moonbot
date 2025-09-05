from __future__ import annotations

import torch
from typing import TYPE_CHECKING
import re

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.assets import Articulation
from isaaclab.utils.math import quat_rotate_inverse, yaw_quat, matrix_from_quat
import numpy as np
import math

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv



def height_termination(
        env, threshold = 0.1, asset_cfg:SceneEntityCfg = SceneEntityCfg("robot"), body_name="base_link"
) -> torch.Tensor:
    if getattr(env.scene.sensors, "height_scanner", None) is not None:
        height_sensor = env.scene.sensors["height_scanner"]
        height = height_sensor.data.pos_w[:, 2].unsqueeze(1) - height_sensor.data.ray_hits_w[..., 2]
        height = height.mean(dim=1)
        return height < threshold
    else:
        robot: Articulation = env.scene[asset_cfg.name]
        body_idx = robot.find_bodies(body_name)[0][0]
        body_height = robot.data.body_pos_w[:, body_idx, 2].squeeze(-1)
        return body_height < threshold
    

def body_illegal_contact(
        env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), threshold = 8.0
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors["contact_forces"]
    body_names = contact_sensor.body_names
    net_contact_forces = contact_sensor.data.net_forces_w

    pattern = re.compile(r'^(?!.*wheel).*leg.*$')

    index = torch.tensor([index for index, name in enumerate(body_names) if pattern.match(name)], device=env.device)

    return torch.any(torch.max(torch.norm(net_contact_forces[:, index], dim=-1), dim=1)[0] > threshold)
    
    
def base_out_of_range(
        env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), threshold = (math.pi)/ 2
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    base_link_posi = asset.data.root_pos_w
    arm_link_1_posi = asset.data.body_state_w[:, asset.find_bodies("Arm_Link1")[0][0], :3]

    # the angle between the vec from base_link to arm_link_1 and the z-axis
    vec = arm_link_1_posi - base_link_posi
    vec = vec / torch.norm(vec, dim=-1, keepdim=True)
    z_axis = torch.tensor([0, 0, 1], device=env.device)
    angle_abs = torch.acos(torch.sum(vec * z_axis, dim=-1))
    return angle_abs > threshold

def link_height_below_minimum(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), minimum_height: float = 0.1
) -> torch.Tensor:
    """Terminate when the asset's root height is below the minimum height.

    Note:
        This is currently only supported for flat terrains, i.e. the minimum height is in the world frame.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    link_height = asset.data.body_pos_w[:, asset.find_bodies("Arm_Link7")[0][0], 2]
    return link_height < minimum_height


def illegal_contact_moonbot(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Terminate when the contact force on the sensor exceeds the force threshold."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    ids = contact_sensor.find_bodies(sensor_cfg.body_names)[0]
    # check if any contact force exceeds the threshold
    return torch.any(
        torch.max(torch.norm(net_contact_forces[:, :, ids], dim=-1), dim=1)[0] > threshold, dim=1
    )
