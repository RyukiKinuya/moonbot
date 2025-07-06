import isaaclab.terrains as terrain_gen

from isaaclab.terrains import TerrainGeneratorCfg
from .mesh_terrains_cfg import MeshMoonSimTerrainCfg


MOONBOT_TERRAIN_CFG = TerrainGeneratorCfg(
    size=(3.2,  12.2),
    curriculum=False,
    num_rows=25,
    num_cols=10,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "moon_sim_terrain": MeshMoonSimTerrainCfg(),
    }
)


