from isaaclab.utils import configclass
from moonbot_envs.envs.agents.rsl_rl_cfg import MoonBotManipulatePPORunnerCfg


@configclass
class MinimalArmReachPPORunnerCfg(MoonBotManipulatePPORunnerCfg):
    experiment_name = "moonbot_minimal_arm_reach"


@configclass
class DragonArmReachPPORunnerCfg(MoonBotManipulatePPORunnerCfg):
    experiment_name = "moonbot_dragon_arm_reach"


@configclass
class TricycleArmReachPPORunnerCfg(MoonBotManipulatePPORunnerCfg):
    experiment_name = "moonbot_tricycle_arm_reach"


@configclass
class MinimalObstacleAvoidPPORunnerCfg(MoonBotManipulatePPORunnerCfg):
    experiment_name = "moonbot_minimal_obstacle_avoid"


@configclass
class DragonObstacleAvoidPPORunnerCfg(MoonBotManipulatePPORunnerCfg):
    experiment_name = "moonbot_dragon_obstacle_avoid"


@configclass
class TricycleObstacleAvoidPPORunnerCfg(MoonBotManipulatePPORunnerCfg):
    experiment_name = "moonbot_tricycle_obstacle_avoid"
