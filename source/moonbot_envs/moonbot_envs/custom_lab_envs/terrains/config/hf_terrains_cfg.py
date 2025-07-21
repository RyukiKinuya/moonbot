from dataclasses import MISSING

from isaaclab.utils import configclass

from moonbot_envs.custom_lab_envs.terrains.terrain_generator_cfg import SubTerrainBaseCfg
from moonbot_envs.custom_lab_envs.terrains.height_field import hf_terrains

@configclass
class HfTerrainBaseCfg(SubTerrainBaseCfg):
    """The base configuration for height field terrains."""

    border_width: float = 0.0
    """The width of the border/padding around the terrain (in m). Defaults to 0.0.

    The border width is subtracted from the :obj:`size` of the terrain. If non-zero, it must be
    greater than or equal to the :obj:`horizontal scale`.
    """
    horizontal_scale: float = 0.1
    """The discretization of the terrain along the x and y axes (in m). Defaults to 0.1."""
    vertical_scale: float = 0.005
    """The discretization of the terrain along the z axis (in m). Defaults to 0.005."""
    slope_threshold: float | None = None
    """The slope threshold above which surfaces are made vertical. Defaults to None,
    in which case no correction is applied."""



@configclass
class HfWaveTerrainCfg(HfTerrainBaseCfg):
    """Configuration for a wave height field terrain."""

    function = hf_terrains.wave_terrain

    amplitude_range: tuple[float, float] = MISSING
    """The minimum and maximum amplitude of the wave (in m)."""
    num_waves: int = 1
    """The number of waves to generate. Defaults to 1.0."""

    # Smoothness control parameters
    smoothness: float = 0.5
    """Terrain smoothness coefficient (0.0-1.0), higher values create smoother terrain"""

    # Wave scale parameters
    wave_scale_range: tuple[float, float] = (0.2, 1.7)
    """Wavelength scale factor range (min, max) as multiple of base wavelength"""

    # Amplitude variation parameters
    amplitude_ratio_range: tuple[float, float] = (0.4, 1.5)
    """Amplitude ratio range (min, max) as fraction of base amplitude"""

    # Noise control parameters
    per_wave_noise: float = 0.05
    """Per-wave noise intensity coefficient"""
    global_noise: float = 0.3
    """Global noise intensity coefficient"""

    # Gaussian blur parameters
    use_gaussian_blur: bool = False
    """Whether to apply Gaussian blur for additional smoothing"""
    blur_sigma_range: tuple[float, float] = (0.5, 3.0)
    """Sigma range (min, max) for Gaussian blur"""


@configclass
class HfOriginWaveTerrainCfg(HfTerrainBaseCfg):
    """Configuration for a wave height field terrain."""

    function =hf_terrains.origin_wave_terrain

    amplitude_range: tuple[float, float] = MISSING
    """The minimum and maximum amplitude of the wave (in m)."""
    num_waves: int = 1.0
    """The number of waves to generate. Defaults to 1.0."""



