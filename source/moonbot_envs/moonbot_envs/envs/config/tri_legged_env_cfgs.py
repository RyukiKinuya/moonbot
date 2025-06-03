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
class TriLeggedManipulateEnvCfg(ManagerBasedRLEnvCfg):
    scene: TriLeggedManipulateSceneCfg = TriLeggedManipulateSceneCfg(num_envs=4096, env_spacing=10) # type: ignore

    observations: ObservationsManipulateCfg = ObservationsManipulateCfg() # type: ignore
    actions: ActionsManipulateCfg = ActionsManipulateCfg() # type: ignore
    commands: TriLeggedCommandsManipulateCfg = TriLeggedCommandsManipulateCfg() # type: ignore

    rewards: RewardsManipulateStandingCfg = RewardsManipulateStandingCfg()
    terminations: TerminationsManipulateCfg = TerminationsManipulateCfg()

    events: EventManipulateCfg = EventManipulateCfg()
    curriculum: CurriculumManipulateCfg = CurriculumManipulateCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 12.0
        self.viewer.eye = (3.5, 3.5, 3.5)

        self.sim.dt = 0.005
        # self.sim.disable_contact_processing = True
        self.observations.policy.enable_corruption = False

        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt


@configclass
class TriLeggedMovingTargetEnvCfg(ManagerBasedRLEnvCfg):
    scene: TriLeggedManipulateSceneCfg = TriLeggedManipulateSceneCfg(num_envs=4096, env_spacing=10)

    observations: ObservationsManipulateCfg = ObservationsManipulateCfg()
    actions: ActionsManipulateCfg = ActionsManipulateCfg()
    commands: TriLeggedMovingTargetCommandCfg = TriLeggedMovingTargetCommandCfg()

    rewards: RewardsManipulateCfg = RewardsManipulateCfg()
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
class TriLeggedTerrianEnvCfg(ManagerBasedRLEnvCfg):
    scene: TriLeggedTerrianSceneCfg = TriLeggedTerrianSceneCfg(num_envs=4096, env_spacing=10)

    observations: TriLeggedHeightScanObservationsCfg = TriLeggedHeightScanObservationsCfg()
    actions: ActionsManipulateCfg = ActionsManipulateCfg()
    commands: TriLeggedMovingTargetCommandCfg = TriLeggedMovingTargetCommandCfg()

    rewards: RewardsManipulateCfg = RewardsManipulateCfg()
    terminations: TerminationsManipulateCfg = TerminationsManipulateCfg()

    events: EventManipulateCfg = EventManipulateCfg()
    curriculum: CurriculumManipulateCfg = CurriculumManipulateCfg()

    def __post_init__(self):
        self.sim.physx.gpu_max_rigid_patch_count = 1000000
        self.decimation = 4
        self.episode_length_s = 20.0
        self.viewer.eye = (3.5, 3.5, 3.5)

        self.sim.dt = 0.005
        # self.sim.disable_contact_processing = True
        self.observations.policy.enable_corruption = False

        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt