from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter
import scipy.interpolate as interpolate
from typing import TYPE_CHECKING

from moonbot_envs.custom_lab_envs.terrains.utils import height_field_to_mesh

if TYPE_CHECKING:
    from moonbot_envs.custom_lab_envs.terrains.config import hf_terrains_cfg


@height_field_to_mesh
def wave_terrain(difficulty: float, cfg: hf_terrains_cfg.HfWaveTerrainCfg) -> np.ndarray:
    # Validate input parameters
    if cfg.num_waves < 0:
        raise ValueError(f"Number of waves must be a positive integer. Got: {cfg.num_waves}.")

    # Calculate terrain dimensions and amplitude
    amplitude = cfg.amplitude_range[0] + difficulty * (cfg.amplitude_range[1] - cfg.amplitude_range[0])
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    amplitude_pixels = int(0.5 * amplitude / cfg.vertical_scale)

    # Create coordinate grid
    x = np.arange(0, width_pixels)
    y = np.arange(0, length_pixels)
    xx, yy = np.meshgrid(x, y, indexing='ij')

    # Adjust wave parameters based on smoothness
    smooth_factor = cfg.smoothness

    # Reduce wave count for smoother terrain
    num_waves_factor = 0.8 + 0.4 * (1 - smooth_factor)
    num_waves = max(1, int(cfg.num_waves * num_waves_factor))

    # Initialize height field
    hf_raw = np.zeros_like(xx, dtype=np.float32)

    # Extract range parameters
    min_wave_scale, max_wave_scale = cfg.wave_scale_range
    min_amp_ratio, max_amp_ratio = cfg.amplitude_ratio_range

    # Generate each wave component
    for _ in range(num_waves):
        # Calculate wavelength scale with smoothness adjustment
        wave_scale = min_wave_scale + smooth_factor * (max_wave_scale - min_wave_scale)
        wave_scale += (1 - smooth_factor) * np.random.rand() * (max_wave_scale - min_wave_scale)

        # Random wave direction and phase
        wave_phase = np.random.rand() * 2 * np.pi
        direction = np.random.rand() * 2 * np.pi
        dir_x, dir_y = np.cos(direction), np.sin(direction)

        # Calculate amplitude with smoothness adjustment
        amp_ratio = min_amp_ratio + smooth_factor * (max_amp_ratio - min_amp_ratio)
        amp_ratio += (1 - smooth_factor) * np.random.rand() * (max_amp_ratio - min_amp_ratio)
        wave_amp = amplitude_pixels * amp_ratio

        # Calculate wave projection
        base_wavelength = length_pixels / cfg.num_waves
        proj_dist = (xx * dir_x + yy * dir_y) * (2 * np.pi / (base_wavelength * wave_scale))
        wave = wave_amp * np.sin(proj_dist + wave_phase)

        # Add per-wave noise (intensity reduced for smoother terrain)
        noise_factor = cfg.per_wave_noise * (1 - smooth_factor)
        noise = noise_factor * wave_amp * (np.random.rand(*xx.shape) - 0.5)
        hf_raw += wave + noise

    # Add global noise (intensity reduced for smoother terrain)
    global_noise_factor = cfg.global_noise * (1 - smooth_factor)
    global_noise = global_noise_factor * difficulty * amplitude_pixels * (np.random.rand(*xx.shape) - 0.5)
    hf_raw += global_noise

    # Apply Gaussian blur for high smoothness levels
    if cfg.use_gaussian_blur and smooth_factor > 0.3:
        from scipy.ndimage import gaussian_filter
        # Calculate sigma based on smoothness (more smoothing at higher values)
        sigma_min, sigma_max = cfg.blur_sigma_range
        sigma = sigma_min + (sigma_max - sigma_min) * smooth_factor
        hf_raw = gaussian_filter(hf_raw, sigma=sigma)

    # Discretize height values
    return np.rint(hf_raw).astype(np.int16)