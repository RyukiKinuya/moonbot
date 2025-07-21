from moonbot_envs.custom_lab_envs.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from .hf_terrains_cfg import HfWaveTerrainCfg, HfOriginWaveTerrainCfg
import isaaclab.terrains as terrain_gen

WAVE_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(100.0, 100.0, 50.0),
    border_width=00.0,
    num_rows=1,
    num_cols=1,
    num_height=3,
    horizontal_scale=0.1,
    vertical_scale=0.05,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=False,
    sub_terrains={
        "plane": terrain_gen.MeshPlaneTerrainCfg(), # type: ignore
        #"random_rough": terrain_gen.HfRandomUniformTerrainCfg(
        #     proportion=0.5,noise_range=(0.01, 0.05), noise_step=0.01, border_width=0.25
        #), # type: ignore
        #"wave": HfOriginWaveTerrainCfg(
        #    proportion=1.0, amplitude_range=(0.1, 0.5), num_waves=1, border_width=0.5,
        #), # type: ignore
        # "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
        #     proportion=0.0, slope_range=(0.0, 0.3), platform_width=2.0, border_width=0.25
        # ),
        # "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
        #     proportion=0.0 ,slope_range=(0.0, 0.3), platform_width=2.0, border_width=0.25
        # ),
    },
)
"""Rough terrains configuration."""
