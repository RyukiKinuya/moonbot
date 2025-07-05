import isaaclab.terrains as terrain_gen

from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg

WAVE_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(50.0, 50.0),
    border_width=00.0,
    num_rows=1,
    num_cols=1,
    num_height=3,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=False,
    sub_terrains={
        # "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
        #     proportion=0.5,noise_range=(0.01, 0.05), noise_step=0.01, border_width=0.25
        # ),
        "wave": terrain_gen.HfWaveTerrainCfg(
            proportion=0.5, amplitude_range=(0.1, 0.5), num_waves=12,border_width=0
        ),
        # "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
        #     proportion=0.0, slope_range=(0.0, 0.3), platform_width=2.0, border_width=0.25
        # ),
        # "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
        #     proportion=0.0 ,slope_range=(0.0, 0.3), platform_width=2.0, border_width=0.25
        # ),
    },
)
"""Rough terrains configuration."""