from dataclasses import MISSING

from isaaclab.utils import configclass

from isaaclab.terrains.height_field.hf_terrains_cfg import HfTerrainBaseCfg
from moonbot_envs.custom_lab_envs.terrains.height_field.hf_terrains import wave_terrain


@configclass
class HfWaveTerrainCfg(HfTerrainBaseCfg):
    """Configuration for a wave height field terrain."""

    function = wave_terrain

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