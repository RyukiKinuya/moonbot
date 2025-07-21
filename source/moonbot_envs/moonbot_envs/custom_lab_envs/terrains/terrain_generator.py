import numpy as np
import os
import torch
import trimesh

import omni.log

from isaaclab.utils.dict import dict_to_md5_hash
from isaaclab.utils.io import dump_yaml
from isaaclab.utils.timer import Timer
from isaaclab.utils.warp import convert_to_warp_mesh

from isaaclab.terrains.height_field import HfTerrainBaseCfg
from .terrain_generator_cfg import FlatPatchSamplingCfg, SubTerrainBaseCfg, TerrainGeneratorCfg
from isaaclab.terrains.trimesh.utils import make_border
from isaaclab.terrains.utils import color_meshes_by_height, find_flat_patches


class TerrainGenerator:
    terrain_mesh: trimesh.Trimesh
    terrain_meshes: list[trimesh.Trimesh]
    """List of trimesh.Trimesh objects for all the generated sub-terrains."""
    terrain_origins: np.ndarray
    """The origin of each sub-terrain. Shape is (num_rows, num_cols, 3)."""
    flat_patches: dict[str, torch.Tensor]

    def __init__(self, cfg: TerrainGeneratorCfg, device: str = "cpu"):
        if len(cfg.sub_terrains) == 0:
            raise ValueError("No sub-terrains specified! Please add at least one sub-terrain.")
        self.cfg = cfg
        self.device = device

        # set common values to all sub-terrains config
        for sub_cfg in self.cfg.sub_terrains.values():
            sub_cfg.size = self.cfg.size
            if isinstance(sub_cfg, HfTerrainBaseCfg):
                sub_cfg.horizontal_scale = self.cfg.horizontal_scale
                sub_cfg.vertical_scale = self.cfg.vertical_scale
                sub_cfg.slope_threshold = self.cfg.slope_threshold

        if self.cfg.use_cache and self.cfg.seed is None:
            omni.log.warn(
                "Cache is enabled but the seed is not set. The terrain generation will not be reproducible."
                " Please set the seed in the terrain generator configuration to make the generation reproducible."
            )

        # if the seed is not set, we assume there is a global seed set and use that.
        # this ensures that the terrain is reproducible if the seed is set at the beginning of the program.
        if self.cfg.seed is not None:
            seed = self.cfg.seed
        else:
            seed = np.random.get_state()[1][0]
        # set the seed for reproducibility
        # note: we create a new random number generator to avoid affecting the global state
        #  in the other places where random numbers are used.
        self.np_rng = np.random.default_rng(seed)

        self.flat_patches = {}
        # create a list of all sub-terrains
        self.terrain_meshes = list()
        self.terrain_origins = np.zeros((self.cfg.num_rows, self.cfg.num_cols ,self.cfg.num_height, 3))

        # parse configuration and add sub-terrains
        # create terrains based on curriculum or randomly
        if self.cfg.curriculum:
            with Timer("[INFO] Generating terrains based on curriculum took"):
                self._generate_curriculum_terrains()
        else:
            with Timer("[INFO] Generating terrains randomly took"):
                self._generate_random_terrains()
        # add a border around the terrains
        self._add_terrain_border()
        self.terrain_mesh = trimesh.util.concatenate(self.terrain_meshes)

        # color the terrain mesh
        if self.cfg.color_scheme == "height":
            self.terrain_mesh = color_meshes_by_height(self.terrain_mesh)
        elif self.cfg.color_scheme == "random":
            self.terrain_mesh.visual.vertex_colors = self.np_rng.choice(
                range(256), size=(len(self.terrain_mesh.vertices), 4)
            )
        elif self.cfg.color_scheme == "none":
            pass
        else:
            raise ValueError(f"Invalid color scheme: {self.cfg.color_scheme}.")

        # offset the entire terrain and origins so that it is centered
        # -- terrain mesh
        transform = np.eye(4)
        transform[:3, -1] = -self.cfg.size[0] * self.cfg.num_rows * 0.5, -self.cfg.size[1] * self.cfg.num_cols * 0.5, -self.cfg.size[2] * self.cfg.num_height * 0.5
        self.terrain_mesh.apply_transform(transform)
        # -- terrain origins
        self.terrain_origins += transform[:3, -1]
        # -- valid patches
        terrain_origins_torch = torch.tensor(self.terrain_origins, dtype=torch.float, device=self.device).unsqueeze(2)
        for name, value in self.flat_patches.items():
            self.flat_patches[name] = value + terrain_origins_torch

    def __str__(self):
        msg = "Terrain Generator:"
        msg += f"\n\tSeed: {self.cfg.seed}"
        msg += f"\n\tNumber of rows: {self.cfg.num_rows}"
        msg += f"\n\tNumber of columns: {self.cfg.num_cols}"
        msg += f"\n\tSub-terrain size: {self.cfg.size}"
        msg += f"\n\tSub-terrain types: {list(self.cfg.sub_terrains.keys())}"
        msg += f"\n\tCurriculum: {self.cfg.curriculum}"
        msg += f"\n\tDifficulty range: {self.cfg.difficulty_range}"
        msg += f"\n\tColor scheme: {self.cfg.color_scheme}"
        msg += f"\n\tUse cache: {self.cfg.use_cache}"
        if self.cfg.use_cache:
            msg += f"\n\tCache directory: {self.cfg.cache_dir}"

        return msg

    def _generate_random_terrains(self):
        """Add terrains based on randomly sampled difficulty parameter."""
        # normalize the proportions of the sub-terrains
        proportions = np.array([sub_cfg.proportion for sub_cfg in self.cfg.sub_terrains.values()])
        proportions /= np.sum(proportions)
        # create a list of all terrain configs
        sub_terrains_cfgs = list(self.cfg.sub_terrains.values())
        # create a buffer to store sub-terrain height using list
        self.sub_terrain_heights = [] 

        # randomly sample sub-terrains
        for index in range(self.cfg.num_rows * self.cfg.num_cols * self.cfg.num_height):
            # coordinate index of the sub-terrain
            (sub_row, sub_col, sub_hig) = np.unravel_index(index, (self.cfg.num_rows, self.cfg.num_cols, self.cfg.num_height))
            # randomly sample terrain index
            sub_index = self.np_rng.choice(len(proportions), p=proportions)
            # randomly sample difficulty parameter
            difficulty = self.np_rng.uniform(*self.cfg.difficulty_range)
            # generate terrain
            mesh, origin, heights = self._get_terrain_mesh(difficulty, sub_terrains_cfgs[sub_index])
            #mesh, origin = self._get_terrain_mesh(difficulty, sub_terrains_cfgs[sub_index])
            self.sub_terrain_heights.append(heights)
            # add to sub-terrains
            self._add_sub_terrain(mesh, origin, sub_row, sub_col,sub_hig, sub_terrains_cfgs[sub_index])

    def _generate_curriculum_terrains(self):
        """Add terrains based on the difficulty parameter."""
        # normalize the proportions of the sub-terrains
        proportions = np.array([sub_cfg.proportion for sub_cfg in self.cfg.sub_terrains.values()])
        proportions /= np.sum(proportions)

        # find the sub-terrain index for each column
        # we generate the terrains based on their proportion (not randomly sampled)
        sub_indices = []
        for index in range(self.cfg.num_cols):
            sub_index = np.min(np.where(index / self.cfg.num_cols + 0.001 < np.cumsum(proportions))[0])
            sub_indices.append(sub_index)
        sub_indices = np.array(sub_indices, dtype=np.int32)
        # create a list of all terrain configs
        sub_terrains_cfgs = list(self.cfg.sub_terrains.values())

        # curriculum-based sub-terrains
        for sub_col in range(self.cfg.num_cols):
            for sub_row in range(self.cfg.num_rows):
                # vary the difficulty parameter linearly over the number of rows
                # note: based on the proportion, multiple columns can have the same sub-terrain type.
                #  Thus to increase the diversity along the rows, we add a small random value to the difficulty.
                #  This ensures that the terrains are not exactly the same. For example, if the
                #  the row index is 2 and the number of rows is 10, the nominal difficulty is 0.2.
                #  We add a small random value to the difficulty to make it between 0.2 and 0.3.
                lower, upper = self.cfg.difficulty_range
                difficulty = (sub_row + self.np_rng.uniform()) / self.cfg.num_rows
                difficulty = lower + (upper - lower) * difficulty
                # generate terrain
                mesh, origin, heights= self._get_terrain_mesh(difficulty, sub_terrains_cfgs[sub_indices[sub_col]])
                # add to sub-terrains
                for sub_hig in range(self.cfg.num_height):
                    self._add_sub_terrain(mesh, origin, sub_row, sub_col, sub_hig, sub_terrains_cfgs[sub_indices[sub_col]])

    """
    Internal helper functions.
    """

    def _add_terrain_border(self):
        """Add a surrounding border over all the sub-terrains into the terrain meshes."""
        # border parameters
        border_size = (
            self.cfg.num_rows * self.cfg.size[0] + 2 * self.cfg.border_width,
            self.cfg.num_cols * self.cfg.size[1] + 2 * self.cfg.border_width,
        )
        inner_size = (self.cfg.num_rows * self.cfg.size[0], self.cfg.num_cols * self.cfg.size[1])
        border_center = (
            self.cfg.num_rows * self.cfg.size[0] / 2,
            self.cfg.num_cols * self.cfg.size[1] / 2,
            -self.cfg.border_height / 2,
        )
        # border mesh
        border_meshes = make_border(border_size, inner_size, height=self.cfg.border_height, position=border_center)
        border = trimesh.util.concatenate(border_meshes)
        # update the faces to have minimal triangles
        selector = ~(np.asarray(border.triangles)[:, :, 2] < -0.1).any(1)
        border.update_faces(selector)
        # add the border to the list of meshes
        self.terrain_meshes.append(border)

    def _add_sub_terrain(
        self, mesh: trimesh.Trimesh, origin: np.ndarray, row: int, col: int, hig:int, sub_terrain_cfg: SubTerrainBaseCfg
    ):
        """Add input sub-terrain to the list of sub-terrains.

        This function adds the input sub-terrain mesh to the list of sub-terrains and updates the origin
        of the sub-terrain in the list of origins. It also samples flat patches if specified.

        Args:
            mesh: The mesh of the sub-terrain.
            origin: The origin of the sub-terrain.
            row: The row index of the sub-terrain.
            col: The column index of the sub-terrain.
        """
        # sample flat patches if specified
        if sub_terrain_cfg.flat_patch_sampling is not None:
            omni.log.info(f"Sampling flat patches for sub-terrain at (row, col):  ({row}, {col})")
            # convert the mesh to warp mesh
            wp_mesh = convert_to_warp_mesh(mesh.vertices, mesh.faces, device=self.device)
            # sample flat patches based on each patch configuration for that sub-terrain
            for name, patch_cfg in sub_terrain_cfg.flat_patch_sampling.items():
                patch_cfg: FlatPatchSamplingCfg
                # create the flat patches tensor (if not already created)
                if name not in self.flat_patches:
                    self.flat_patches[name] = torch.zeros(
                        (self.cfg.num_rows, self.cfg.num_cols, patch_cfg.num_patches, 3), device=self.device
                    )
                # add the flat patches to the tensor
                self.flat_patches[name][row, col] = find_flat_patches(
                    wp_mesh=wp_mesh,
                    origin=origin,
                    num_patches=patch_cfg.num_patches,
                    patch_radius=patch_cfg.patch_radius,
                    x_range=patch_cfg.x_range,
                    y_range=patch_cfg.y_range,
                    z_range=patch_cfg.z_range,
                    max_height_diff=patch_cfg.max_height_diff,
                )

        # transform the mesh to the correct position
        transform = np.eye(4)
        transform[0:2, -1] = (row + 0.5) * self.cfg.size[0], (col + 0.5) * self.cfg.size[1]
        transform[2, -1] = (hig + 0.5) * self.cfg.size[2]
        mesh.apply_transform(transform)
        # add mesh to the list
        self.terrain_meshes.append(mesh)
        # add origin to the list
        self.terrain_origins[row, col, hig] = origin + transform[:3, -1]


    def _get_terrain_mesh(self, difficulty: float, cfg: SubTerrainBaseCfg) -> tuple[trimesh.Trimesh, np.ndarray, np.ndarray]:
        cfg = cfg.copy()
        cfg.difficulty = float(difficulty)
        cfg.seed = self.cfg.seed
        # generate hash for the sub-terrain
        sub_terrain_hash = dict_to_md5_hash(cfg.to_dict())
        # generate the file name
        sub_terrain_cache_dir = os.path.join(self.cfg.cache_dir, sub_terrain_hash)
        sub_terrain_obj_filename = os.path.join(sub_terrain_cache_dir, "mesh.obj")
        sub_terrain_csv_filename = os.path.join(sub_terrain_cache_dir, "origin.csv")
        sub_terrain_meta_filename = os.path.join(sub_terrain_cache_dir, "cfg.yaml")

        # check if hash exists - if true, load the mesh and origin and return
        if self.cfg.use_cache and os.path.exists(sub_terrain_obj_filename):
            # load existing mesh
            mesh = trimesh.load_mesh(sub_terrain_obj_filename, process=False)
            origin = np.loadtxt(sub_terrain_csv_filename, delimiter=",")
            # return the generated mesh
            return mesh, origin
        meshes, origin, heights= cfg.function(difficulty, cfg) # type: ignore
        # meshes, origin = cfg.function(difficulty, cfg) # type: ignore
        mesh = trimesh.util.concatenate(meshes)
        # offset mesh such that they are in their center
        transform = np.eye(4)
        transform[0:2, -1] = -cfg.size[0] * 0.5, -cfg.size[1] * 0.5
        mesh.apply_transform(transform)
        # change origin to be in the center of the sub-terrain
        origin += transform[0:3, -1]

        # if caching is enabled, save the mesh and origin
        if self.cfg.use_cache:
            # create the cache directory
            os.makedirs(sub_terrain_cache_dir, exist_ok=True)
            # save the data
            mesh.export(sub_terrain_obj_filename)
            np.savetxt(sub_terrain_csv_filename, origin, delimiter=",", header="x,y,z")
            dump_yaml(sub_terrain_meta_filename, cfg)
        return mesh, origin, heights


