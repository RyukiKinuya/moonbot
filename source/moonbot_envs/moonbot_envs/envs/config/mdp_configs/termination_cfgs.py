from __future__ import annotations

import math
from dataclasses import MISSING

from isaaclab.managers import TerminationTermCfg as DoneTerm
import moonbot_envs.moonbot.mdp as mdp
from isaaclab.managers import SceneEntityCfg

from isaaclab.utils import configclass

@configclass
class TerminationsManipulateCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base_link"), "threshold": 8.0},
    )

@configclass
class UniLeggedTerminationsManipulateCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="Arm.*"), "threshold": 8.0},
    )


@configclass
class UniLeggedUnlimitedReachingManipulateTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # base_contact = DoneTerm(
    #     func=mdp.illegal_contact,
    #     params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="Arm_Link"), "threshold": 8.0},
    # )

@configclass
class DragonLocomotionTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="leg4link[3-4]|leg3link[3-6]|leg3gripper2|leg3gripper2_straight"), "threshold": 8.0},
    )

@configclass
class DragonReachingTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces",  body_names="leg4link[3-4]|leg3link[5-6]|leg3gripper2|leg3gripper2_straight"), "threshold": 8.0},
    )