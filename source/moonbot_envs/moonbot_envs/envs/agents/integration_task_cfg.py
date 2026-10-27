"""M2oE runner configs for simultaneous three-morphology task training."""
from isaaclab.utils import configclass
from moonbot_envs.envs.agents.M2oE_cfg import M2oE_Cfg


@configclass
class IntegrationArmReachM2oECfg(M2oE_Cfg):
    max_num_modules = 3
    experiment_name = "MoonBot_Integration_ArmReach"


@configclass
class IntegrationObstacleAvoidM2oECfg(M2oE_Cfg):
    max_num_modules = 3
    experiment_name = "MoonBot_Integration_ObstacleAvoid"
