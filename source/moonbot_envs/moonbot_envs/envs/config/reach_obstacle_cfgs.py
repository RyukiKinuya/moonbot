"""Arm-reaching and obstacle-avoidance Gym tasks for the MoonBot assets.

Install this module in ``moonbot_envs`` and import ``register_tasks`` from the
package's envs/__init__.py. The task terms adapt the cited papers to MoonBot's
body/joint names; they are not a verbatim reproduction of either benchmark.
"""
from __future__ import annotations

import torch
import gymnasium as gym

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import (ObservationGroupCfg, ObservationTermCfg as ObsTerm,
                               RewardTermCfg as RewTerm, SceneEntityCfg,
                               TerminationTermCfg as DoneTerm)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from moonbot_envs.envs import mdp
from moonbot_envs.envs.config.sence_cfgs import (
    UniLeggedManipulateSceneCfg, DragonReachingSceneCfg, TriLeggedManipulateSceneCfg,
)


EE_NAMES = {
    "minimal": "Arm_Link7",
    "dragon": "leg3gripper2_base",
    "tricycle": "gripper_palm",
}


def ee_goal_error(env, ee_name: str) -> torch.Tensor:
    """Euclidean EE-to-goal distance in the environment-local frame."""
    robot = env.scene["robot"]
    body_id = robot.find_bodies(ee_name)[0][0]
    current = robot.data.body_pos_w[:, body_id] - env.scene.env_origins
    command = env.command_manager.get_command("ee_pose")[:, :3]
    return torch.linalg.vector_norm(current - command, dim=-1)


def reach_dense_reward(env, ee_name: str, std: float = 0.15) -> torch.Tensor:
    # Distance shaping follows the end-effector distance term in Alessi et al. (2024).
    return torch.exp(-ee_goal_error(env, ee_name) / std)


def reach_success(env, ee_name: str, tolerance: float = 0.06) -> torch.Tensor:
    return (ee_goal_error(env, ee_name) < tolerance).float()


def obstacle_ranges(env, ee_name: str) -> torch.Tensor:
    """Relative coordinates of the three fixed box obstacles (compact state)."""
    # Adapted obstacle-relative state representation; Tang et al. (Journal of Sensors,
    # 2022, doi:10.1155/2022/4194747) use obstacle-position state and collision/progress
    # reward terms in a wheeled-robot task, rather than this manipulator's exact vector.
    # Boxes are at these local coordinates in every environment (see obstacle scenes).
    centers = torch.tensor(((0.65, -0.25, 0.95), (0.90, 0.0, 1.05),
                            (0.65, 0.25, 0.95)), device=env.device, dtype=torch.float32)
    robot = env.scene["robot"]
    ee_id = robot.find_bodies(ee_name)[0][0]
    ee = robot.data.body_pos_w[:, ee_id, :] - env.scene.env_origins
    rel = centers.unsqueeze(0) - ee.unsqueeze(1)
    return rel.reshape(env.num_envs, -1)


def obstacle_clearance_penalty(env, ee_name: str, margin: float = 0.18) -> torch.Tensor:
    # Potential-field safety penalty; contact is handled separately as a hard penalty.
    distances = torch.linalg.vector_norm(obstacle_ranges(env, ee_name).reshape(env.num_envs, 3, 3), dim=-1)
    return torch.square(torch.clamp(margin - distances, min=0.0)).sum(dim=-1)


def contact_collision(env, sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
                       threshold: float = 12.0) -> torch.Tensor:
    sensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w
    return (torch.linalg.vector_norm(forces, dim=-1).amax(dim=-1) > threshold).float()


@configclass
class ObstacleSceneMixin:
    # Static collision geometry, shared across the six obstacle-avoidance variants.
    obstacle_left = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/ObstacleLeft",
        spawn=sim_utils.CuboidCfg(size=(0.16, 0.16, 0.65),
                                  rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                                  collision_props=sim_utils.CollisionPropertiesCfg()),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.65, -0.25, 0.325)),
    )
    obstacle_center = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/ObstacleCenter",
        spawn=sim_utils.CuboidCfg(size=(0.16, 0.16, 0.75),
                                  rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                                  collision_props=sim_utils.CollisionPropertiesCfg()),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.90, 0.0, 0.375)),
    )
    obstacle_right = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/ObstacleRight",
        spawn=sim_utils.CuboidCfg(size=(0.16, 0.16, 0.65),
                                  rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                                  collision_props=sim_utils.CollisionPropertiesCfg()),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.65, 0.25, 0.325)),
    )


@configclass
class MinimalObstacleSceneCfg(ObstacleSceneMixin, UniLeggedManipulateSceneCfg):
    pass


@configclass
class DragonObstacleSceneCfg(ObstacleSceneMixin, DragonReachingSceneCfg):
    pass


@configclass
class TricycleObstacleSceneCfg(ObstacleSceneMixin, TriLeggedManipulateSceneCfg):
    pass


def _terms(ee_name: str, with_obstacles: bool):
    """Return manager terms; keeping this factory avoids morphology-specific reward drift."""
    terms = {
        "reach": RewTerm(func=reach_dense_reward, weight=2.0,
                         params={"ee_name": ee_name, "std": 0.15}),
        "success": RewTerm(func=reach_success, weight=10.0,
                            params={"ee_name": ee_name, "tolerance": 0.06}),
        "action_rate": RewTerm(func=mdp.action_rate_l2, weight=-0.002),
        "joint_velocity": RewTerm(func=mdp.joint_vel_l2, weight=-0.0001,
                                  params={"asset_cfg": SceneEntityCfg("robot")}),
    }
    if with_obstacles:
        terms["clearance"] = RewTerm(func=obstacle_clearance_penalty, weight=-4.0,
                                     params={"ee_name": ee_name})
        terms["collision"] = RewTerm(func=contact_collision, weight=-20.0,
                                     params={"sensor_cfg": SceneEntityCfg("contact_forces"), "threshold": 12.0})
    annotations = {name: RewTerm for name in terms}
    return configclass(type("RewardsCfg", (), {"__annotations__": annotations, **terms}))


def _observations(ee_name: str, with_obstacles: bool):
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot")})
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot")})
        goal = ObsTerm(func=mdp.ee_pose_command, params={"command_name": "ee_pose"})
        ee = ObsTerm(func=mdp.ee_pose, params={"ee_name": ee_name})
        if with_obstacles:
            obstacle_relative_positions = ObsTerm(func=obstacle_ranges, params={"ee_name": ee_name})

        def __post_init__(self):
            self.concatenate_terms = True
            self.enable_corruption = False
    @configclass
    class ObsCfg:
        policy: PolicyCfg = PolicyCfg()
    return ObsCfg


def _make_cfg(scene_cfg, action_cfg, command_cfg, ee_name: str, with_obstacles: bool):
    obs_cfg = _observations(ee_name, with_obstacles)
    rewards = _terms(ee_name, with_obstacles)

    @configclass
    class TerminationsCfg:
        time_out = DoneTerm(func=mdp.time_out, time_out=True)
        success = DoneTerm(func=reach_success, params={"ee_name": ee_name, "tolerance": 0.06})

    def post_init(self):
        self.decimation = 4
        self.episode_length_s = 8.0
        self.sim.dt = 0.005
        self.viewer.eye = (3.0, 3.0, 2.5)
        if with_obstacles:
            self.scene.contact_forces.update_period = self.sim.dt

    # Keep local config classes as concrete runtime types for configclass.
    namespace = {
        "__annotations__": {
            "scene": scene_cfg, "observations": obs_cfg,
            "actions": action_cfg, "commands": command_cfg,
            "rewards": type(rewards), "terminations": TerminationsCfg,
        },
        "scene": scene_cfg(num_envs=256, env_spacing=4.0),
        "observations": obs_cfg(), "actions": action_cfg(),
        "commands": command_cfg(), "rewards": rewards,
        "terminations": TerminationsCfg(), "__post_init__": post_init,
    }
    return configclass(type("MoonBotTaskEnvCfg", (ManagerBasedRLEnvCfg,), namespace))


from moonbot_envs.envs.config.mdp_configs.command_cfgs import (
    UniLeggedCommandsManipulateCfg, DragonReachingCommandCfg,
    TriLeggedCommandsManipulateCfg,
)
from moonbot_envs.envs.config.mdp_configs.action_cfgs import (
    UniLeggedActionsManipulateCfg, DragonActionsCfg, ActionsManipulateCfg,
)
MINIMAL_REACH_CFG = _make_cfg(UniLeggedManipulateSceneCfg, UniLeggedActionsManipulateCfg,
                              UniLeggedCommandsManipulateCfg, EE_NAMES["minimal"], False)
DRAGON_REACH_CFG = _make_cfg(DragonReachingSceneCfg, DragonActionsCfg,
                             DragonReachingCommandCfg, EE_NAMES["dragon"], False)
TRICYCLE_REACH_CFG = _make_cfg(TriLeggedManipulateSceneCfg, ActionsManipulateCfg,
                               TriLeggedCommandsManipulateCfg, EE_NAMES["tricycle"], False)
MINIMAL_OBSTACLE_CFG = _make_cfg(MinimalObstacleSceneCfg, UniLeggedActionsManipulateCfg,
                                 UniLeggedCommandsManipulateCfg, EE_NAMES["minimal"], True)
DRAGON_OBSTACLE_CFG = _make_cfg(DragonObstacleSceneCfg, DragonActionsCfg,
                                DragonReachingCommandCfg, EE_NAMES["dragon"], True)
TRICYCLE_OBSTACLE_CFG = _make_cfg(TricycleObstacleSceneCfg, ActionsManipulateCfg,
                                  TriLeggedCommandsManipulateCfg, EE_NAMES["tricycle"], True)
