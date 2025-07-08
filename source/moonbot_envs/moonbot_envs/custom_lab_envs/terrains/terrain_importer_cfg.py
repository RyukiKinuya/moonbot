# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING, Literal

import isaaclab.sim as sim_utils
from isaaclab.utils import configclass

from .terrain_importer import TerrainImporter

if TYPE_CHECKING:
    from .terrain_generator_cfg import TerrainGeneratorCfg


@configclass
class SubTerrainBaseCfg:
    """Base class for terrain configurations.

    All the sub-terrain configurations must inherit from this class.

    The :attr:`size` attribute is the size of the generated sub-terrain. Based on this, the terrain must
    extend from :math:`(0, 0)` to :math:`(size[0], size[1])`.
    """

    function: Callable[[float, SubTerrainBaseCfg], tuple[list[trimesh.Trimesh], np.ndarray, np.ndarray]] = MISSING
    """Function to generate the terrain.

    This function must take as input the terrain difficulty and the configuration parameters and
    return a tuple with a list of ``trimesh`` mesh objects and the terrain origin.
    """

    proportion: float = 1.0
    """Proportion of the terrain to generate. Defaults to 1.0.

    This is used to generate a mix of terrains. The proportion corresponds to the probability of sampling
    the particular terrain. For example, if there are two terrains, A and B, with proportions 0.3 and 0.7,
    respectively, then the probability of sampling terrain A is 0.3 and the probability of sampling terrain B
    is 0.7.
    """

    size: tuple[float, float] = (10.0, 10.0)
    """The width (along x) and length (along y) of the terrain (in m). Defaults to (10.0, 10.0).

    In case the :class:`~isaaclab.terrains.TerrainImporterCfg` is used, this parameter gets overridden by
    :attr:`isaaclab.scene.TerrainImporterCfg.size` attribute.
    """

    flat_patch_sampling: dict[str, FlatPatchSamplingCfg] | None = None
    """Dictionary of configurations for sampling flat patches on the sub-terrain. Defaults to None,
    in which case no flat patch sampling is performed.

    The keys correspond to the name of the flat patch sampling configuration and the values are the
    corresponding configurations.
    """



@configclass
class TerrainImporterCfg:
    """Configuration for the terrain manager."""

    class_type: type = TerrainImporter
    """The class to use for the terrain importer.

    Defaults to :class:`isaaclab.terrains.terrain_importer.TerrainImporter`.
    """

    collision_group: int = -1
    """The collision group of the terrain. Defaults to -1."""

    prim_path: str = MISSING
    """The absolute path of the USD terrain prim.

    All sub-terrains are imported relative to this prim path.
    """

    num_envs: int = 1
    """The number of environment origins to consider. Defaults to 1.

    In case, the :class:`~isaaclab.scene.InteractiveSceneCfg` is used, this parameter gets overridden by
    :attr:`isaaclab.scene.InteractiveSceneCfg.num_envs` attribute.
    """

    terrain_type: Literal["generator", "plane", "usd"] = "generator"
    """The type of terrain to generate. Defaults to "generator".

    Available options are "plane", "usd", and "generator".
    """

    terrain_generator: TerrainGeneratorCfg | None = None
    """The terrain generator configuration.

    Only used if ``terrain_type`` is set to "generator".
    """

    usd_path: str | None = None
    """The path to the USD file containing the terrain.

    Only used if ``terrain_type`` is set to "usd".
    """

    env_spacing: float | None = None
    """The spacing between environment origins when defined in a grid. Defaults to None.

    Note:
      This parameter is used only when the ``terrain_type`` is "plane" or "usd".
    """

    visual_material: sim_utils.VisualMaterialCfg | None = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 0.0))
    """The visual material of the terrain. Defaults to a dark gray color material.

    This parameter is used for both the "generator" and "plane" terrains.

    - If the ``terrain_type`` is "generator", then the material is created at the path
      ``{prim_path}/visualMaterial`` and applied to all the sub-terrains.
    - If the ``terrain_type`` is "plane", then the diffuse color of the material is set to
      to the grid color of the imported ground plane.
    """

    physics_material: sim_utils.RigidBodyMaterialCfg = sim_utils.RigidBodyMaterialCfg()
    """The physics material of the terrain. Defaults to a default physics material.

    The material is created at the path: ``{prim_path}/physicsMaterial``.

    .. note::
        This parameter is used only when the ``terrain_type`` is "generator" or "plane".
    """

    max_init_terrain_level: int | None = None
    """The maximum initial terrain level for defining environment origins. Defaults to None.

    The terrain levels are specified by the number of rows in the grid arrangement of
    sub-terrains. If None, then the initial terrain level is set to the maximum
    terrain level available (``num_rows - 1``).

    Note:
      This parameter is used only when sub-terrain origins are defined.
    """

    debug_vis: bool = False
    """Whether to enable visualization of terrain origins for the terrain. Defaults to False."""
