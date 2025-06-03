from __future__ import annotations

from isaaclab.managers import CurriculumTermCfg as CurrTerm
import moonbot_envs.moonbot.mdp as mdp

from isaaclab.utils import configclass



@configclass
class CurriculumManipulateCfg:
    pass
    # action_rate = CurrTerm(
    #     func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -0.005, "num_steps": 12000}
    # )

    # joint_vel = CurrTerm(
    #     func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -0.001, "num_steps": 12000}
    # )

    # terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)

    


@configclass
class UniLeggedCurriculumManipulateCfg:
    # stage 1 
    # modify_gravity = CurrTerm(
    #     func=mdp.modify_scene_gravity, params={"gravity": (0.0, 0.0, -9.85004196167), "num_steps": 0}
    # )

    # stage 2
    # diff_from_init_pose = CurrTerm(
    #     func=mdp.modify_reward_weight, params={"term_name": "diff_from_init_pose", "weight": 0.0, "num_steps": 0}
    # )

    # base_balance = CurrTerm(
    #     func=mdp.modify_reward_weight, params={"term_name": "base_balance", "weight": -2.0, "num_steps": 0}
    # )

    # base_dis_from_origin = CurrTerm(
    #     func=mdp.modify_reward_weight, params={"term_name": "base_dis_from_origin", "weight": -1.0, "num_steps": 0}
    # )

    # base_vel = CurrTerm(
    #     func=mdp.modify_reward_weight, params={"term_name": "base_vel", "weight": -0.5, "num_steps": 0}
    # )

    is_alive = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "is_alive", "weight": 20.0, "num_steps": 0}
    )

    # stage 3
    action_rate = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -0.001, "num_steps": 500}
    )

    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -0.001, "num_steps": 500}
    )

    ee_position = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "end_effector_position_tracking", "weight": -4.0, "num_steps": 500}
    )

    ee_position_fine_grained = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "end_effecstor_position_tracking_fine_grained", "weight": 2.0, "num_steps": 500}
    )

    ee_position_fine_grained = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "end_effector_orientation_tracking", "weight": -2.0, "num_steps": 500}
    )


@configclass
class UniLeggedUnlimitedCurriculumCfg:
    pass

@configclass
class DragonCurriculumCfg:
    pass