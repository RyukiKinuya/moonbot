from __future__ import annotations

import torch
import math
from dataclasses import MISSING


from isaaclab.managers.action_manager import ActionTermCfg as ActionTerm
import omni.isaac.contrib_tasks.moonbot.mdp as mdp

from isaaclab.utils import configclass

@configclass
class ActionsManipulateCfg:
    arm_action: ActionTerm = mdp.JointPositionActionCfg(asset_name="robot", joint_names=["arm_joint.*"], scale=0.5, use_default_offset=True)
    # leg_action: ActionTerm = mdp.JointPositionActionCfg(asset_name="robot", joint_names=["(?!.*wheel.*)leg.*"], scale=0.5, use_default_offset=True)
    wheel_action: ActionTerm = mdp.JointVelocityActionCfg(asset_name="robot", joint_names=[".*wheel.*"], scale=50.0)


@configclass
class UniLeggedActionsManipulateCfg:
    leg_action: ActionTerm = mdp.JointPositionActionCfg(asset_name="robot", joint_names=["joint.*"], scale=0.5, use_default_offset=True)
    wheel_action: ActionTerm = mdp.JointVelocityActionCfg(asset_name="robot", joint_names=["Wheel.*"], scale=10.0)

@configclass
class DragonActionsCfg:
    connect_link_action: ActionTerm = mdp.JointPositionActionCfg(asset_name="robot", joint_names=["leg4joint[1-3,7]"], scale=0.1, use_default_offset=True)
    arm_action: ActionTerm = mdp.JointPositionActionCfg(asset_name="robot", joint_names=["leg3joint.*"], scale=0.1, use_default_offset=True)
    # wheel_action1: ActionTerm = mdp.WheelVelocityActionCfg(asset_name="robot", joint_names=["wheel12.*joint"], scale=50.0)
    # wheel_action2: ActionTerm = mdp.WheelVelocityActionCfg(asset_name="robot", joint_names=["wheel14.*joint"], scale=50.0)
    wheel_action: ActionTerm = mdp.JointVelocityActionCfg(asset_name="robot", joint_names=["wheel.*joint"], scale=10.0)
    
    