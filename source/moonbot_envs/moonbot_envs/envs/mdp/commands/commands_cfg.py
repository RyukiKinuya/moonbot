import math
from dataclasses import MISSING

from isaaclab.managers import CommandTermCfg
from isaaclab.utils import configclass

from .command import EePoseCommand, BaseHeightCommand, EePoseMovingCommand

@configclass
class EePoseCommandCfg(CommandTermCfg):
    class_type: type = EePoseCommand

    asset_name: str = MISSING
    ee_name: str = MISSING

    @configclass
    class Ranges:
        """Uniform distribution ranges for the pose commands."""

        dis_xy: tuple[float, float] = MISSING  # min max [m]
        pos_z: tuple[float, float] = MISSING  # min max [m]
        roll: tuple[float, float] = MISSING  # min max [rad]
        pitch: tuple[float, float] = MISSING  # min max [rad]
        yaw: tuple[float, float] = MISSING  # min max [rad]

    ranges: Ranges = MISSING
    """Ranges for the commands."""


@configclass
class BaseHeightCommandCfg(CommandTermCfg):
    class_type: type = BaseHeightCommand
    
    asset_name: str = MISSING

    @configclass
    class Ranges:
        base_height: tuple[float, float] = MISSING

    ranges: Ranges = MISSING

 
@configclass
class EePoseMovingCommandCfg(CommandTermCfg):
    class_type: type = EePoseMovingCommand

    asset_name: str = MISSING
    ee_name: str = MISSING

    @configclass
    class Ranges:
        start_point: tuple[float, float, float] = MISSING
        end_point: tuple[float, float, float] = MISSING
        roll: tuple[float, float] = MISSING
        pitch: tuple[float, float] = MISSING
        yaw: tuple[float, float] = MISSING

    ranges: Ranges = MISSING
