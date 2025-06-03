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
class UniLeggedManipulateEnvCfg(ManagerBasedRLEnvCfg):
    scene: UniLeggedManipulateSceneCfg = UniLeggedManipulateSceneCfg(num_envs=4096, env_spacing=2)

    observations: UniLeggedObservationManipulateCfg = UniLeggedObservationManipulateCfg()
    actions: UniLeggedActionsManipulateCfg = UniLeggedActionsManipulateCfg()
    commands: UniLeggedCommandsManipulateCfg = UniLeggedCommandsManipulateCfg()

    rewards: UniLeggedManipulateRewardsCfg = UniLeggedManipulateRewardsCfg()
    terminations: UniLeggedTerminationsManipulateCfg = UniLeggedTerminationsManipulateCfg()

    events: UniLeggedEventManipulateCfg = UniLeggedEventManipulateCfg()
    curriculum: UniLeggedCurriculumManipulateCfg = UniLeggedCurriculumManipulateCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 6.0
        self.viewer.eye = (3.5, 3.5, 3.5)

        self.sim.dt = 0.005

        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt


@configclass
class UniLeggedFixBaseReachingEnvCfg(ManagerBasedRLEnvCfg):
    scene: UniLeggedFixBaseReachingSceneCfg = UniLeggedFixBaseReachingSceneCfg(num_envs=4096, env_spacing=4)

    observations: ObservationsManipulateCfg = ObservationsManipulateCfg()
    actions: UniLeggedActionsManipulateCfg = UniLeggedActionsManipulateCfg()
    commands: RewardsManipulateCfg = RewardsManipulateCfg()

    rewards: UniLeggedManipulateRewardsCfg = UniLeggedManipulateRewardsCfg()
    terminations: TerminationsManipulateCfg = TerminationsManipulateCfg()

    events: EventManipulateCfg = EventManipulateCfg()
    curriculum: CurriculumManipulateCfg = CurriculumManipulateCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 6.0
        self.viewer.eye = (3.5, 3.5, 3.5)

        self.sim.dt = 0.005

        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt


@configclass
class UniLeggedStandingReachingEnvCfg(ManagerBasedRLEnvCfg):
    scene: UniLeggedStandingReachingSceneCfg = UniLeggedStandingReachingSceneCfg(num_envs=4096, env_spacing=2)

    observations: UniLeggedObservationManipulateCfg = UniLeggedObservationManipulateCfg()
    actions: UniLeggedActionsManipulateCfg = UniLeggedActionsManipulateCfg()
    commands: UniLeggedCommandsManipulateCfg = UniLeggedCommandsManipulateCfg()

    rewards: UniLeggedStandingReachingRewardsCfg = UniLeggedStandingReachingRewardsCfg()
    terminations: UniLeggedTerminationsManipulateCfg = UniLeggedTerminationsManipulateCfg()

    events: UniLeggedEventManipulateCfg = UniLeggedEventManipulateCfg()
    curriculum: UniLeggedCurriculumManipulateCfg = UniLeggedCurriculumManipulateCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 6.0
        self.viewer.eye = (3.5, 3.5, 3.5)

        self.sim.dt = 0.005

        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt


@configclass
class UniLeggedUnlimitedReachingEnvCfg(ManagerBasedRLEnvCfg):
    scene: UniLeggedUnlimitedReachingSceneCfg = UniLeggedUnlimitedReachingSceneCfg(num_envs=4096, env_spacing=2)

    observations: UniLeggedUnlimitedObservationManipulateCfg = UniLeggedUnlimitedObservationManipulateCfg()
    actions: UniLeggedActionsManipulateCfg = UniLeggedActionsManipulateCfg()
    commands: UniLeggedUnlimitedCommandsCfg = UniLeggedUnlimitedCommandsCfg()

    rewards: UniLeggedUnlimitedReachingRewardsCfg = UniLeggedUnlimitedReachingRewardsCfg()
    terminations: UniLeggedUnlimitedReachingManipulateTerminationsCfg = UniLeggedUnlimitedReachingManipulateTerminationsCfg()

    events: UniLeggedUnlimitedManipulateEventCfg = UniLeggedUnlimitedManipulateEventCfg()
    curriculum: UniLeggedUnlimitedCurriculumCfg = UniLeggedUnlimitedCurriculumCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 6.0
        self.viewer.eye = (3.5, 3.5, 3.5)

        self.sim.dt = 0.005

        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt


@configclass
class UniLeggedWalkingReachingEnvCfg(ManagerBasedRLEnvCfg):
    scene: UniLeggedStandingReachingSceneCfg = UniLeggedStandingReachingSceneCfg(num_envs=4096, env_spacing=5.0)

    observations: UniLeggedObservationManipulateCfg = UniLeggedObservationManipulateCfg()
    actions: UniLeggedActionsManipulateCfg = UniLeggedActionsManipulateCfg()
    commands: UniLeggedMovingTargetCommandCfg = UniLeggedMovingTargetCommandCfg()

    rewards: UniLeggedStandingReachingRewardsCfg = UniLeggedStandingReachingRewardsCfg()
    terminations: UniLeggedUnlimitedReachingManipulateTerminationsCfg = UniLeggedUnlimitedReachingManipulateTerminationsCfg()

    events: UniLeggedEventManipulateCfg = UniLeggedEventManipulateCfg()
    curriculum: UniLeggedCurriculumManipulateCfg = UniLeggedCurriculumManipulateCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 6.0
        self.viewer.eye = (3.5, 3.5, 3.5)

        self.sim.dt = 0.005

        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

