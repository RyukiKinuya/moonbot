from __future__ import annotations

import numpy as np
import scipy.spatial.transform as tf
import torch
import trimesh
from trimesh.transformations import translation_matrix, rotation_matrix, compose_matrix
from typing import TYPE_CHECKING

from isaaclab.terrains.trimesh.utils import *
from isaaclab.terrains.trimesh.utils import make_plane

if TYPE_CHECKING:
    from . import mesh_terrains_cfg


def moon_sim_terrain(
        diffculty: float, cfg: mesh_terrains_cfg.MeshMoonSimTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    origin = np.array([cfg.size[0] / 2.0, 1.0, 0.0])
    # slope_height = cfg.slope_height * diffculty
    meshes_list = list()

    # plane
    plane_mesh = make_plane(cfg.size, 0.0, center_zero=False)
    meshes_list.append(plane_mesh)

    # border
    if cfg.border_width > 0.0:
        border_center = [0.5 * cfg.size[0], 0.5 * cfg.size[1], 0]
        border_inner_size = (cfg.size[0] - 2 * cfg.border_width, cfg.size[1] - 2 * cfg.border_width)
        make_borders = make_border(cfg.size, border_inner_size, 0, border_center)
        # add the border meshes to the list of meshes
        meshes_list += make_borders

    sphere = trimesh.creation.icosphere(subdivisions=3, radius=cfg.slope_radius)
    half_sphere = sphere.slice_plane(plane_origin=[0.0, 0.0, 0.0], plane_normal=[1.0, 0.0, 0.0])
    
    if half_sphere is not None: 
        # first sphere
        translation = translation_matrix([0.0 + cfg.border_width, (cfg.size[1] - 2*cfg.border_width)/4, cfg.slope_height-cfg.slope_radius])
        rotation = rotation_matrix(0, [0, 0, 1])
        first_sphere = half_sphere.copy().apply_transform(translation @ rotation)
        # change the color of the first sphere to blue
        first_sphere.visual.face_colors = np.array([0, 0, 255, 255])
        meshes_list.append(first_sphere)

        # second sphere
        translation = translation_matrix([cfg.size[0] - cfg.border_width, (cfg.size[1] - 2*cfg.border_width)*3/4, cfg.slope_height-cfg.slope_radius])
        rotation = rotation_matrix(np.radians(180), [0, 0, 1])
        second_sphere = half_sphere.copy().apply_transform(translation @ rotation)
        meshes_list.append(second_sphere)

    # walls
    wall_height = 1.5
    wall_width = 0.1
    
    # left wall
    translation = translation_matrix([-wall_width/2, cfg.size[1]/2, wall_height/2])
    rotation = rotation_matrix(0, [0, 0, 1])
    left_wall = trimesh.creation.box([wall_width, cfg.size[1], wall_height], translation @ rotation)
    meshes_list.append(left_wall)

    # right wall
    translation = translation_matrix([cfg.size[0]-wall_width/2, cfg.size[1]/2, wall_height/2])
    rotation = rotation_matrix(0, [0, 0, 1])
    right_wall = trimesh.creation.box([wall_width, cfg.size[1], wall_height], translation @ rotation)
    meshes_list.append(right_wall)  

    # back wall
    translation = translation_matrix([cfg.size[0]/2, cfg.size[1] + wall_width/2, wall_height/2])
    rotation = rotation_matrix(0, [0, 0, 1])
    back_wall = trimesh.creation.box([cfg.size[0], wall_width, wall_height], translation @ rotation)
    meshes_list.append(back_wall)

    # front wall
    translation = translation_matrix([cfg.size[0]/2, -wall_width/2, wall_height/2])
    rotation = rotation_matrix(0, [0, 0, 1])
    front_wall = trimesh.creation.box([cfg.size[0], wall_width, wall_height], translation @ rotation)
    meshes_list.append(front_wall)

    return meshes_list, origin
