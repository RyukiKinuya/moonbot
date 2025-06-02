from dataclasses import MISSING
from typing import Literal

from .mesh_terrains import *
from isaaclab.utils import configclass

from isaaclab.terrains.terrain_generator_cfg import SubTerrainBaseCfg

@configclass
class MeshMoonSimTerrainCfg(SubTerrainBaseCfg):
    function = moon_sim_terrain

    border_width: float = 0.1
    slope_radius: float = 2.5
    slope_height: float = 0.2
