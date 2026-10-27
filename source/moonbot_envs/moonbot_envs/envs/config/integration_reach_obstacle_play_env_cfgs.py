"""Play configurations for the integrated Arm Reach and Obstacle Avoid tasks."""
from moonbot_envs.envs.config.Integration_play_env_cfgs import (
    PlayIntegrationEnvCfg, PlayIntegrationSceneCfg,
)
from moonbot_envs.envs.config.integration_reach_obstacle_cfgs import _make_cfg

PlayIntegrationArmReachEnvCfg = _make_cfg(
    play=True, play_env_cfg=PlayIntegrationEnvCfg, play_scene_cfg=PlayIntegrationSceneCfg,
)
PlayIntegrationObstacleAvoidEnvCfg = _make_cfg(
    obstacles=True, play=True, play_env_cfg=PlayIntegrationEnvCfg,
    play_scene_cfg=PlayIntegrationSceneCfg,
)
