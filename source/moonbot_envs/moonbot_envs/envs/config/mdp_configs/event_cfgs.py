from __future__ import annotations

import torch
import math
from dataclasses import MISSING

import moonbot_envs.moonbot.mdp as mdp
from isaaclab.managers import EventTermCfg as EventTerm


from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.utils import configclass

@configclass
class EventManipulateCfg:
    reset_robot_joints = EventTerm(
        func=mdp.reset_scene_to_default,
        mode="reset",
    )

    randomize_physic_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 1.2),
            "dynamic_friction_range": (0.6, 0.8),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        }
    )

    randomize_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "mass_distribution_params": (20, 20),
            "operation": "add",
            "distribution": "uniform",
        }
    )


@configclass
class UniLeggedEventManipulateCfg:
    reset_robot_joints = EventTerm(
        func=mdp.reset_scene_to_default,
        mode="reset",
    )

    randomize_physic_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 1.0),
            "dynamic_friction_range": (0.6, 0.8),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        }
    )

    randomize_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "mass_distribution_params": (2000, 2000),
            "operation": "add",
            "distribution": "uniform",
        }
    )

    randomize_link_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="Arm_Link.*"),
            "mass_distribution_params": (0.1, 0.1),
            "operation": "scale",
            "distribution": "uniform",
        }
    )


@configclass
class UniLeggedUnlimitedManipulateEventCfg:
    reset_robot_joints = EventTerm(
        func=mdp.reset_scene_to_default,
        mode="reset",
    )

    randomize_physic_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 1.0),
            "dynamic_friction_range": (0.6, 0.8),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        }
    )

@configclass
class DragonLocomotionEventCfg:
    reset_robot_joints = EventTerm(
        func=mdp.reset_scene_to_default,
        mode="reset",
    )

    randomize_physic_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 1.0),
            "dynamic_friction_range": (0.6, 0.8),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        }
    )


@configclass
class DragonReachingEventCfg:
    reset_robot_joints = EventTerm(
        func=mdp.reset_scene_to_default,
        mode="reset",
    )

    randomize_physic_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 1.2),
            "dynamic_friction_range": (0.6, 0.8),
            "restitution_range": (0.9, 1.0),
            "num_buckets": 64,
        }
    )

    randomize_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link|wheel14_body"),
            "mass_distribution_params": (100, 200),
            "operation": "add",
            "distribution": "uniform",
        }
    )