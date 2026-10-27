"""Registration for the research environments added by the specialized M2oE study.

This module is deliberately separate from the repository's original
``moonbot_envs.envs`` registration file.  Specialized train/play entry points
import it explicitly, so the original package files remain unchanged.
"""
from __future__ import annotations

import gymnasium as gym

from moonbot_envs.envs import agents
from moonbot_envs.envs.agents.integration_task_cfg import (
    IntegrationArmReachM2oECfg, IntegrationObstacleAvoidM2oECfg,
)
from moonbot_envs.envs.agents.reach_obstacle_cfg import (
    MinimalArmReachPPORunnerCfg, DragonArmReachPPORunnerCfg,
    TricycleArmReachPPORunnerCfg, MinimalObstacleAvoidPPORunnerCfg,
    DragonObstacleAvoidPPORunnerCfg, TricycleObstacleAvoidPPORunnerCfg,
)
from moonbot_envs.envs.config.integration_reach_obstacle_cfgs import (
    IntegrationArmReachEnvCfg, IntegrationObstacleAvoidEnvCfg,
)
from moonbot_envs.envs.config.integration_reach_obstacle_play_env_cfgs import (
    PlayIntegrationArmReachEnvCfg, PlayIntegrationObstacleAvoidEnvCfg,
)
from moonbot_envs.envs.config.reach_obstacle_cfgs import (
    MINIMAL_REACH_CFG, DRAGON_REACH_CFG, TRICYCLE_REACH_CFG,
    MINIMAL_OBSTACLE_CFG, DRAGON_OBSTACLE_CFG, TRICYCLE_OBSTACLE_CFG,
)
from moonbot_envs.envs.config.reach_obstacle_play_env_cfgs import (
    MINIMAL_REACH_PLAY_CFG, DRAGON_REACH_PLAY_CFG, TRICYCLE_REACH_PLAY_CFG,
    MINIMAL_OBSTACLE_PLAY_CFG, DRAGON_OBSTACLE_PLAY_CFG, TRICYCLE_OBSTACLE_PLAY_CFG,
)


def _register(task_id, entry_point, cfg, runner_cfg, *, integration=False):
    if task_id in gym.registry:
        return
    kwargs = {"env_cfg_entry_point": cfg}
    if integration:
        kwargs["M2oE_entry_point"] = runner_cfg
    else:
        kwargs["rsl_rl_cfg_entry_point"] = runner_cfg
    gym.register(id=task_id, entry_point=entry_point, disable_env_checker=True, kwargs=kwargs)


def register_specialized_tasks() -> None:
    """Register only the environments used by the new research scripts."""
    standard = (
        ("MoonBot-Minimal-ArmReach-v0", MINIMAL_REACH_CFG, MinimalArmReachPPORunnerCfg),
        ("MoonBot-Dragon-ArmReach-v0", DRAGON_REACH_CFG, DragonArmReachPPORunnerCfg),
        ("MoonBot-Tricycle-ArmReach-v0", TRICYCLE_REACH_CFG, TricycleArmReachPPORunnerCfg),
        ("MoonBot-Minimal-ObstacleAvoid-v0", MINIMAL_OBSTACLE_CFG, MinimalObstacleAvoidPPORunnerCfg),
        ("MoonBot-Dragon-ObstacleAvoid-v0", DRAGON_OBSTACLE_CFG, DragonObstacleAvoidPPORunnerCfg),
        ("MoonBot-Tricycle-ObstacleAvoid-v0", TRICYCLE_OBSTACLE_CFG, TricycleObstacleAvoidPPORunnerCfg),
        ("MoonBot-Minimal-ArmReach-Play-v0", MINIMAL_REACH_PLAY_CFG, MinimalArmReachPPORunnerCfg),
        ("MoonBot-Dragon-ArmReach-Play-v0", DRAGON_REACH_PLAY_CFG, DragonArmReachPPORunnerCfg),
        ("MoonBot-Tricycle-ArmReach-Play-v0", TRICYCLE_REACH_PLAY_CFG, TricycleArmReachPPORunnerCfg),
        ("MoonBot-Minimal-ObstacleAvoid-Play-v0", MINIMAL_OBSTACLE_PLAY_CFG, MinimalObstacleAvoidPPORunnerCfg),
        ("MoonBot-Dragon-ObstacleAvoid-Play-v0", DRAGON_OBSTACLE_PLAY_CFG, DragonObstacleAvoidPPORunnerCfg),
        ("MoonBot-Tricycle-ObstacleAvoid-Play-v0", TRICYCLE_OBSTACLE_PLAY_CFG, TricycleObstacleAvoidPPORunnerCfg),
    )
    for task_id, cfg, runner_cfg in standard:
        _register(task_id, "isaaclab.envs:ManagerBasedRLEnv", cfg, runner_cfg)

    integration = (
        ("Integration_ArmReach_v1", IntegrationArmReachEnvCfg, IntegrationArmReachM2oECfg),
        ("Integration_ObstacleAvoid_v1", IntegrationObstacleAvoidEnvCfg, IntegrationObstacleAvoidM2oECfg),
        ("Integration_ArmReach_Play_v1", PlayIntegrationArmReachEnvCfg, IntegrationArmReachM2oECfg),
        ("Integration_ObstacleAvoid_Play_v1", PlayIntegrationObstacleAvoidEnvCfg, IntegrationObstacleAvoidM2oECfg),
    )
    for task_id, cfg, runner_cfg in integration:
        _register(task_id, "moonbot_envs.custom_lab_envs:CustomManagerBasedRLEnv",
                  cfg, runner_cfg, integration=True)


register_specialized_tasks()
