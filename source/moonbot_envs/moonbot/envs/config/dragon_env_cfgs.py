from __future__ import annotations

from isaaclab.envs import ManagerBasedRLEnvCfg

from isaaclab.utils import configclass
from .mdp_configs.observation_cfgs import *
from .mdp_configs.action_cfgs import *
from .mdp_configs.command_cfgs import *
from .mdp_configs.reward_cfgs import *
from .mdp_configs.termination_cfgs import *
from .mdp_configs.event_cfgs import *
from .mdp_configs.currculum_cfgs import *
from .sence_cfgs import *

@configclass
class DragonLocomotionEnvCfg(ManagerBasedRLEnvCfg):
    scene: DragonLocomotionSceneCfg = DragonLocomotionSceneCfg(num_envs=4096, env_spacing=10)

    observations: DragonLocomotionObservationCfg = DragonLocomotionObservationCfg()
    actions: DragonActionsCfg = DragonActionsCfg()
    commands: DragonLocomotionCommandCfg = DragonLocomotionCommandCfg()

    rewards: DragonLocomotionRewardsCfg = DragonLocomotionRewardsCfg()
    terminations: DragonLocomotionTerminationsCfg = DragonLocomotionTerminationsCfg()

    events: DragonLocomotionEventCfg = DragonLocomotionEventCfg()
    curriculum: DragonCurriculumCfg = DragonCurriculumCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0
        self.viewer.eye = (3.5, 3.5, 3.5)
        
        self.sim.dt = 0.005

        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt


@configclass
class DragonReachingEnvCfg(ManagerBasedRLEnvCfg):
    scene: DragonReachingSceneCfg = DragonReachingSceneCfg(num_envs=4096, env_spacing=10)

    observations: DragonObservationCfg = DragonObservationCfg()
    actions: DragonActionsCfg = DragonActionsCfg()
    commands: DragonReachingCommandCfg = DragonReachingCommandCfg()

    rewards: DragonReachingRewardsCfg = DragonReachingRewardsCfg()
    terminations: DragonReachingTerminationsCfg = DragonReachingTerminationsCfg()

    events: DragonReachingEventCfg = DragonReachingEventCfg()
    curriculum: DragonCurriculumCfg = DragonCurriculumCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 6.0
        self.viewer.eye = (3.5, 3.5, 3.5)
        
        self.sim.dt = 0.005

        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt