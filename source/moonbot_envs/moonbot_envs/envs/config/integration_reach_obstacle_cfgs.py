"""Multi-morphology integration tasks: Minimal, Dragon and Tricycle train together."""
from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm, SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from M2oE.configs import morphology_configs
from moonbot_envs.envs import mdp
from moonbot_envs.envs.config.Integration_env_cfgs import IntegrationEnvCfg, IntegrationSceneCfg, IntegrationTerminationCfg
from moonbot_envs.envs.mdp.integration_task_mdp import (
    module_task_obs, reach_progress, reach_success, obstacle_clearance, obstacle_collision, all_morphologies_reached,
)

EE_NAMES = {"moonbot_minimal": "Arm_Link7", "moonbot_dragon": "gripper_palm", "moonbot_full": "gripper_palm"}


@configclass
class ReachSceneCfg(IntegrationSceneCfg):
    """Use the repository's integrated scene with all three robot assets."""


@configclass
class ObstacleSceneCfg(ReachSceneCfg):
    obstacle_left: AssetBaseCfg = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/ObstacleLeft",
        spawn=sim_utils.CuboidCfg(size=(0.16, 0.16, 0.65), collision_props=sim_utils.CollisionPropertiesCfg()),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.65, -0.25, 0.325)))
    obstacle_center: AssetBaseCfg = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/ObstacleCenter",
        spawn=sim_utils.CuboidCfg(size=(0.16, 0.16, 0.75), collision_props=sim_utils.CollisionPropertiesCfg()),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.90, 0.0, 0.375)))
    obstacle_right: AssetBaseCfg = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/ObstacleRight",
        spawn=sim_utils.CuboidCfg(size=(0.16, 0.16, 0.65), collision_props=sim_utils.CollisionPropertiesCfg()),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.65, 0.25, 0.325)))


def _observations(with_obstacles: bool):
    @configclass
    class MorphObs(ObservationGroupCfg):
        def __init__(self, asset: str, count: int):
            super().__init__()
            for i in range(count):
                setattr(self, f"module_{i}", ObsTerm(func=module_task_obs, params={
                    "asset_cfg": SceneEntityCfg(asset), "module_no": i,
                    "command_name": f"ee_pose_{asset}", "end_effector_name": EE_NAMES[asset],
                    "with_obstacles": with_obstacles}))
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class GlobalObs(ObservationGroupCfg):
        def __init__(self, asset: str):
            super().__init__()
            cfg = SceneEntityCfg(asset)
            self.base_lin_vel = ObsTerm(func=mdp.base_lin_vel, params={"asset_cfg": cfg})
            self.base_ang_vel = ObsTerm(func=mdp.base_ang_vel, params={"asset_cfg": cfg})
            self.projected_gravity = ObsTerm(func=mdp.projected_gravity, params={"asset_cfg": cfg})
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    class ObsCfg:
        __annotations__ = {}
    for asset, count, suffix in zip(morphology_configs.morphology_list, (1, 2, 3), ("minimal", "dragon", "full")):
        ObsCfg.__annotations__[f"obs_{suffix}"] = MorphObs
        ObsCfg.__annotations__[f"global_obs_{asset}"] = GlobalObs
        setattr(ObsCfg, f"obs_{suffix}", MorphObs(asset, count))
        setattr(ObsCfg, f"global_obs_{asset}", GlobalObs(asset))
    return configclass(ObsCfg)


def _commands():
    class CommandsCfg:
        __annotations__ = {}
    for asset in morphology_configs.morphology_list:
        CommandsCfg.__annotations__[f"ee_pose_{asset}"] = mdp.EePoseCommandCfg
        setattr(CommandsCfg, f"ee_pose_{asset}", mdp.EePoseCommandCfg(
            asset_name=asset, ee_name=EE_NAMES[asset], resampling_time_range=(4.0, 4.0), debug_vis=True,
            ranges=mdp.EePoseCommandCfg.Ranges(dis_xy=(0.5, 0.9), pos_z=(0.7, 1.25),
                roll=(-math.pi, math.pi), pitch=(-math.pi, math.pi), yaw=(-math.pi, math.pi))))
    return configclass(CommandsCfg)


def _rewards(with_obstacles: bool):
    @configclass
    class RewardGroup:
        def __init__(self, asset: str):
            cfg = SceneEntityCfg(asset)
            params = {"asset_cfg": cfg, "command_name": f"ee_pose_{asset}", "end_effector_name": EE_NAMES[asset]}
            self.reach_progress = RewTerm(func=reach_progress, weight=2.0, params=params)
            self.reach_success = RewTerm(func=reach_success, weight=10.0, params={**params, "tolerance": 0.06})
            self.action_rate = RewTerm(func=mdp.action_rate_modular, weight=-0.01, params={"asset_cfg": cfg})
            self.joint_velocity = RewTerm(func=mdp.joint_vel_l2, weight=-0.0001, params={"asset_cfg": cfg})
            if with_obstacles:
                self.obstacle_clearance = RewTerm(func=obstacle_clearance, weight=-4.0, params=params)
                self.obstacle_collision = RewTerm(func=obstacle_collision, weight=-20.0,
                    params={"sensor_cfg": SceneEntityCfg(f"contact_forces_{asset}"), "threshold": 12.0})
    class RewardsCfg:
        __annotations__ = {}
    for asset in morphology_configs.morphology_list:
        RewardsCfg.__annotations__[f"reward_{asset}"] = RewardGroup
        setattr(RewardsCfg, f"reward_{asset}", RewardGroup(asset))
    return configclass(RewardsCfg)


def _make_cfg(obstacles: bool = False, play: bool = False, play_env_cfg=None, play_scene_cfg=None):
    if play and (play_env_cfg is None or play_scene_cfg is None):
        raise ValueError("play_env_cfg and play_scene_cfg are required for Play configs")
    base = play_env_cfg if play else IntegrationEnvCfg
    obs_cfg, reward_cfg, commands_cfg = _observations(obstacles), _rewards(obstacles), _commands()
    scene_type = ObstacleSceneCfg if obstacles else (play_scene_cfg if play else ReachSceneCfg)
    @configclass
    class TaskTerminationsCfg(IntegrationTerminationCfg):
        all_morphologies_reached: DoneTerm = DoneTerm(func=all_morphologies_reached, time_out=False)
    @configclass
    class TaskCfg(base):
        observations = obs_cfg()
        commands = commands_cfg()
        rewards = reward_cfg()
        terminations = TaskTerminationsCfg()
        scene = scene_type(num_envs=1 if play else 1024, env_spacing=5.0)
        def __post_init__(self):
            super().__post_init__()
            self.episode_length_s = 12.0
            self.viewer.asset_name = "moonbot_full"
            if play:
                for group in ("obs_minimal", "obs_dragon", "obs_full", "global_obs_moonbot_minimal",
                              "global_obs_moonbot_dragon", "global_obs_moonbot_full"):
                    getattr(self.observations, group).enable_corruption = False
                for asset in morphology_configs.morphology_list:
                    getattr(self.commands, f"ee_pose_{asset}").debug_vis = False
    return configclass(TaskCfg)


IntegrationArmReachEnvCfg = _make_cfg()
IntegrationObstacleAvoidEnvCfg = _make_cfg(obstacles=True)
