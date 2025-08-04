import gymnasium as gym

from .config.tri_legged_env_cfgs import *
from .config.uni_legged_env_cfgs import *
from .config.dragon_env_cfgs import *
from .config.Integration_env_cfgs import IntegrationEnvCfg
from . import agents

gym.register(
    id="Integration_Locomotion_v1",
    entry_point="moonbot_envs.custom_lab_envs:CustomManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": IntegrationEnvCfg,
        # "M2oE_entry_point": agents.M2oE_cfg.M2oE_Cfg
        "M2oE_entry_point": agents.Transformer_cfg.Transformer_Cfg
    }
)

gym.register(
    id="Tri_Legged_Manipulate_v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": TriLeggedManipulateEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.MoonBotManipulatePPORunnerCfg
    },
)

gym.register(
    id="Tri_Legged_Moving_Target_v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": TriLeggedMovingTargetEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.MoonBotManipulatePPORunnerCfg
    },
)

gym.register(
    id="Uni_Legged_Manipulate_v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": UniLeggedManipulateEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.MoonBotManipulatePPORunnerCfg
    },
)

gym.register(
    id="Uni_Legged_Standing_Reaching_v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": UniLeggedStandingReachingEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.MoonBotUniLeggedStandingReachingPPORunnerCfg
    },
)

gym.register(
    id="Uni_Legged_Fix_Base_Reaching_v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": UniLeggedFixBaseReachingEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.MoonBotUniLeggedManipulatePPORunnerCfg,
    },
)

gym.register(
    id="Uni_Legged_Unlimited_Reaching_v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": UniLeggedUnlimitedReachingEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.MoonBotUniLeggedManipulatePPORunnerCfg,
    },
)

gym.register(
    id="Uni_Legged_Walking_Reaching_v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": UniLeggedWalkingReachingEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.MoonBotUniLeggedManipulatePPORunnerCfg,
    },
)

gym.register(
    id="Tri_Legged_Terrain_Traverse_v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": TriLeggedTerrianEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.MoonBotManipulatePPORunnerCfg,
    },
)

gym.register(
    id="Dragon_Locomotion_v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": DragonLocomotionEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.MoonBotLocomotionPPORunnerCfg,
    },
)

gym.register(
    id="Dragon_Reaching_v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": DragonReachingEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.MoonBotLocomotionPPORunnerCfg,
    },
)
