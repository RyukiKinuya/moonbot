"""Play configurations for the six single-morphology research tasks."""
from isaaclab.utils import configclass
from moonbot_envs.envs.config.reach_obstacle_cfgs import (
    MINIMAL_REACH_CFG, DRAGON_REACH_CFG, TRICYCLE_REACH_CFG,
    MINIMAL_OBSTACLE_CFG, DRAGON_OBSTACLE_CFG, TRICYCLE_OBSTACLE_CFG,
)


def _make_play_cfg(base_cfg):
    def post_init(self):
        super(type(self), self).__post_init__()
        self.scene.num_envs = 1
        self.observations.policy.enable_corruption = False
        self.commands.ee_pose.debug_vis = False
    return configclass(type(f"{base_cfg.__name__}Play", (base_cfg,), {"__post_init__": post_init}))


MINIMAL_REACH_PLAY_CFG = _make_play_cfg(MINIMAL_REACH_CFG)
DRAGON_REACH_PLAY_CFG = _make_play_cfg(DRAGON_REACH_CFG)
TRICYCLE_REACH_PLAY_CFG = _make_play_cfg(TRICYCLE_REACH_CFG)
MINIMAL_OBSTACLE_PLAY_CFG = _make_play_cfg(MINIMAL_OBSTACLE_CFG)
DRAGON_OBSTACLE_PLAY_CFG = _make_play_cfg(DRAGON_OBSTACLE_CFG)
TRICYCLE_OBSTACLE_PLAY_CFG = _make_play_cfg(TRICYCLE_OBSTACLE_CFG)
