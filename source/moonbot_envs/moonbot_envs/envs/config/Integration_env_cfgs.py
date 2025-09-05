from __future__ import annotations

import math
from typing import TYPE_CHECKING

from M2oE.configs import morphology_configs
from isaaclab.managers.manager_term_cfg import CurriculumTermCfg
from isaaclab.terrains import terrain_generator
import moonbot_envs.envs.mdp as mdp
from moonbot_envs.assets import *
from moonbot_envs.custom_lab_envs.manager_term_cfg import ActionGroupCfg, RewardGroupCfg
from moonbot_envs.custom_lab_envs.terrains.config.wave_terrains_cfg import WAVE_TERRAINS_CFG
from moonbot_envs.custom_lab_envs.terrains.terrain_importer_cfg import TerrainImporterCfg
# from isaaclab.terrains import TerrainImporterCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.envs.common import ViewerCfg

@configclass
class IntegrationSceneCfg(InteractiveSceneCfg):
    def __init__(self, num_envs: int = 4096, env_spacing: float = 10.0):
        super().__init__(num_envs=num_envs, env_spacing=env_spacing)
        import M2oE.configs.morphology_configs as morphology_configs

        # init ground and skey light
        setattr(self, "ground", TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=WAVE_TERRAINS_CFG,
            max_init_terrain_level= 1,
            collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.1,
                dynamic_friction=0.8,
            ),
            visual_material=sim_utils.MdlFileCfg(
                mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
                project_uvw=True,
                texture_scale=(0.25, 0.25),
            ),
            debug_vis=False,
        ))

        setattr(self, "light", AssetBaseCfg(
            prim_path="/World/skyLight",
            spawn=sim_utils.DomeLightCfg(
                intensity=750.0,
                texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
            ),
        ))

        for asset_name in morphology_configs.morphology_list:
            # asset config
            setattr(self, f"{asset_name}", morphology_configs.morphology_asset_cfg[asset_name]
                .replace(prim_path="{ENV_REGEX_NS}/"+asset_name).replace(
                    init_state= ArticulationCfg.InitialStateCfg(
                        pos=morphology_configs.morphology_asset_init_state[asset_name]["base_position"],
                        joint_pos=morphology_configs.morphology_asset_init_state[asset_name]["joint_pos"],
                )
            ))

            # contact_sensor config
            setattr(self, f"contact_forces_{asset_name}", ContactSensorCfg(
                prim_path="{ENV_REGEX_NS}/" + asset_name + "/" + morphology_configs.contact_sensor_links[asset_name],
                history_length=1,
                track_air_time=True,
                force_threshold=8.0,
            ))


@configclass
class IntegrationObsCfg:
    @configclass
    class MoonbotObsCfg(ObsGroup):
        def __init__(self, asset_cfg: SceneEntityCfg, num_morphologies: int ):
            super().__init__()
            for i in range(num_morphologies):
                module_name = f"module_{i}"
                setattr(self, module_name, ObsTerm(func=mdp.module_obs, params={"asset_cfg": asset_cfg, "module_no": i}))

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class GlobalCfg(ObsGroup):
        def __init__(self, asset_cfg: SceneEntityCfg = SceneEntityCfg("moonbot_minimal")):
            super().__init__()
            # self.base_height = ObsTerm(func=mdp.base_height_obs, params={"asset_cfg": asset_cfg})
            self.base_lin_vel = ObsTerm(func=mdp.base_lin_vel, params={"asset_cfg": asset_cfg})
            self.base_ang_vel = ObsTerm(func=mdp.base_ang_vel, params={"asset_cfg": asset_cfg})
            self.velocity_commands = ObsTerm(
                func=mdp.generated_commands,
                params={"command_name": f"base_velocity_{asset_cfg.name}"},
            )
            self.projected_gravity = ObsTerm(
                func=mdp.projected_gravity,
                params={"asset_cfg": asset_cfg},
            )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    obs_minimal: MoonbotObsCfg = MoonbotObsCfg(asset_cfg=SceneEntityCfg("moonbot_minimal"), num_morphologies=1)
    obs_dragon:  MoonbotObsCfg = MoonbotObsCfg(asset_cfg=SceneEntityCfg("moonbot_dragon"), num_morphologies=2)
    obs_full:    MoonbotObsCfg = MoonbotObsCfg(asset_cfg=SceneEntityCfg("moonbot_full"), num_morphologies=3)
    global_obs_moonbot_minimal: GlobalCfg = GlobalCfg(asset_cfg=SceneEntityCfg("moonbot_minimal"))
    global_obs_moonbot_dragon:  GlobalCfg = GlobalCfg(asset_cfg=SceneEntityCfg("moonbot_dragon"))
    global_obs_moonbot_full:    GlobalCfg = GlobalCfg(asset_cfg=SceneEntityCfg("moonbot_full"))

@configclass
class IntegrationActCfg:
    @configclass
    class MoonbotActCfg(ActionGroupCfg):
        def __init__(self, asset_name: str, num_morphologies: int = 1):
            super().__init__()
            for i in range(num_morphologies):
                leg_action_name = f"module_{i}_action_leg"
                leg_action_term = mdp.JointPositionActionCfg(
                    asset_name=asset_name,
                    joint_names=morphology_configs.joint_names_dict[asset_name][i]["leg"],
                    scale= 0.1,
                    use_default_offset=True,
                    preserve_order=True,
                )
                setattr(self, leg_action_name, leg_action_term)

                wheel_action_name = f"module_{i}_action_wheel"
                wheel_action_term = mdp.JointVelocityActionCfg(
                    asset_name=asset_name,
                    joint_names=morphology_configs.joint_names_dict[asset_name][i]["wheel"],
                    scale= 10,
                    preserve_order=True,
                )
                setattr(self, wheel_action_name, wheel_action_term)

    act_moonbot_minimal: MoonbotActCfg = MoonbotActCfg(
        asset_name="moonbot_minimal",
        num_morphologies=1
    )
    act_moonbot_dragon: MoonbotActCfg = MoonbotActCfg(
        asset_name="moonbot_dragon",
        num_morphologies=2
    )
    act_moonbot_full: MoonbotActCfg = MoonbotActCfg(
        asset_name="moonbot_full",
        num_morphologies=3
    )

@configclass
class IntegrationCmdCfg:
    def __init__(self):
        for asset_name in morphology_configs.morphology_list:
            setattr(self, f"base_velocity_{asset_name}", mdp.UniformVelocityCommandCfg(
                asset_name=asset_name,
                resampling_time_range=(10.0, 10.0),
                rel_standing_envs=0.1,
                rel_heading_envs=1.0,
                heading_command=True,
                heading_control_stiffness=0.5,
                debug_vis=True,
                ranges=mdp.UniformVelocityCommandCfg.Ranges(
                    lin_vel_x=(-1.2, 1.2), lin_vel_y=(-0.0, 0.0), ang_vel_z=(-math.pi/3, math.pi/3), heading=(-math.pi, math.pi)
                ),
            ))


@configclass
class IntegrationRewardCfg:
    @configclass
    class MoonbotRewardCfg(RewardGroupCfg):
        def __init__(self, asset_cfg: SceneEntityCfg):
            self.track_lin_vel_xy_exp = RewTerm(
                func=mdp.track_lin_vel_xy_exp, 
                weight=6.0, params={"asset_cfg": asset_cfg, "command_name": f"base_velocity_{asset_cfg.name}", "std": math.sqrt(0.25)}
            )

            self.track_ang_vel_z_exp = RewTerm(
                func=mdp.track_ang_vel_z_exp, 
                weight=3.0, params={"asset_cfg": asset_cfg, "command_name": f"base_velocity_{asset_cfg.name}", "std": math.sqrt(0.25)}
            )            
            
            self.is_alive = RewTerm(
                func=mdp.is_alive,
                weight=0.0,
            )

            self.diff_from_init_pose = RewTerm(
                func=mdp.diff_from_init_pose,
                weight=5.0,
                params={"asset_cfg": asset_cfg},
            )

            # self.wheel_ang_vel = RewTerm(
            #     func=mdp.wheel_joint_ang_velocity_reward,
            #     weight=-0.1,
            #     params={"asset_cfg": asset_cfg},
            # )

            #self.wheel_on_ground = RewTerm(
            #    func=mdp.wheel_on_ground,
            #    weight=-1.0,
            #    params={"asset_cfg": asset_cfg, "contact_sensor_cfg": SceneEntityCfg(f"contact_forces_{asset_cfg.name}")},
            #)

            self.wheel_rolling_resistance = RewTerm(
                func=mdp.wheel_rolling_consistency,
                weight = -1.0,
                params={"asset_cfg": asset_cfg, "command_name": f"base_velocity_{asset_cfg.name}"},
            )

            if asset_cfg.name == "moonbot_full":
                self.wheel_distance = RewTerm(
                    func=mdp.wheel_distances,
                    weight= 1.0,
                    params={
                        "asset_cfg": asset_cfg,
                    },
                )

            # if asset_cfg.name != "moonbot_minimal":
            #     self.wheel_orientation = RewTerm(
            #         func=mdp.bad_wheel_orientation,
            #         weight= 1.0,
            #         params={
            #             "asset_cfg": asset_cfg,
            #         },
            #     )

            self.base_balance = RewTerm(
                func=mdp.base_balance,
                weight=-5.0,
                params={"asset_cfg": asset_cfg},
            )

            self.undesired_contacts = RewTerm(
                func=mdp.undesired_contacts_moonbot,
                weight=-0.5,
                params={"sensor_cfg": SceneEntityCfg(f"contact_forces_{asset_cfg.name}"), "threshold": 1.0},
            )
            self.lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0, params={"asset_cfg": asset_cfg})
            
            self.ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.1, params={"asset_cfg": asset_cfg})
        
            self.dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2e-5, params={"asset_cfg": asset_cfg})
            
            self.power = RewTerm(func=mdp.joint_power, weight=-5e-4, params={"asset_cfg": asset_cfg})
            
            self.dof_vel_l2 = RewTerm(
                func=mdp.joint_vel_l2, weight=-0.02, params={"asset_cfg": asset_cfg}
            )

            self.action_rate = RewTerm(func=mdp.action_rate_modular, weight=-0.01, params={"asset_cfg": asset_cfg})


    reward_moonbot_minimal: MoonbotRewardCfg = MoonbotRewardCfg(asset_cfg=SceneEntityCfg("moonbot_minimal", joint_names=morphology_configs.joint_names_dict_flat["moonbot_minimal"]["leg"]))
    reward_moonbot_dragon: MoonbotRewardCfg = MoonbotRewardCfg(asset_cfg=SceneEntityCfg("moonbot_dragon", joint_names=morphology_configs.joint_names_dict_flat["moonbot_dragon"]["leg"]))
    reward_moonbot_full: MoonbotRewardCfg = MoonbotRewardCfg(asset_cfg=SceneEntityCfg("moonbot_full", joint_names=morphology_configs.joint_names_dict_flat["moonbot_full"]["leg"]))


@configclass
class IntegrationTerminationCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact_minimal = DoneTerm(
        func=mdp.illegal_contact_moonbot,
        params={"sensor_cfg": SceneEntityCfg("contact_forces_moonbot_minimal", body_names="base_link"), "threshold": 1.0},
    )
    base_contact_dragon = DoneTerm(
        func=mdp.illegal_contact_moonbot,
        params={"sensor_cfg": SceneEntityCfg("contact_forces_moonbot_dragon", body_names="leg4link[3-4]|leg3link[3-6]|leg3gripper2|leg3gripper2_straight"), "threshold": 1.0},
    )
    base_contact_full = DoneTerm(
        func=mdp.illegal_contact_moonbot,
        params={"sensor_cfg": SceneEntityCfg("contact_forces_moonbot_full", body_names="base"), "threshold": 1.0}
    )

@configclass
class IntegrationEventCfg:
    def __init__(self):
        setattr(self, "reset_base", EventTerm(
            func=mdp.reset_root_state_random,
            mode="reset",
            params={
            "pose_range": {"z": (0.05, 0.10),"yaw": (-3.14, 3.14)},
            "velocity_range": {
                # Narrow ranges to keep initial state close to standing pose
                "x": (-0.05, 0.05),
                "y": (-0.05, 0.05),
                "z": (-0.02, 0.02),
                "roll": (-0.1, 0.1),
                "pitch": (-0.1, 0.1),
                "yaw": (-0.1, 0.1),
                },
            },
        ))
        import M2oE.configs.morphology_configs as morphology_configs

        for asset_name in morphology_configs.morphology_list:
            setattr(self, f"wheel_physics_randomize_{asset_name}", EventTerm(
                func=mdp.randomize_rigid_body_material,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg(asset_name, body_names=morphology_configs.wheel_link_name_dict[asset_name]),
                    "static_friction_range": (0.8, 1.2),
                    "dynamic_friction_range": (0.4, 0.9),
                    "restitution_range": (0.0, 0.0),
                    "num_buckets": 16
                },
            ))

            setattr(self, f"base_mass_randomize_{asset_name}", EventTerm(
                func=mdp.randomize_rigid_body_mass,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg(asset_name, body_names=morphology_configs.base_link_name_dict[asset_name]),
                    "mass_distribution_params": (0.8, 1.2),
                    "operation": "scale",
                },
            ))

            #setattr(self, f"base_external_force_{asset_name}", EventTerm(
            #    func=mdp.apply_external_force_torque,
            #    mode="reset",
            #    params={
            #        "asset_cfg": SceneEntityCfg(asset_name, body_names=morphology_configs.base_link_name_dict[asset_name]),
            #        "force_range": (-50.0, 50.0),
            #        "torque_range": (-10.0, 10.0),
            #    },
            #))

            setattr(self, f"reset_joints_{asset_name}", EventTerm(
                func=mdp.reset_joints_by_offset,
                mode="reset",
                params={
                    "asset_cfg": SceneEntityCfg(asset_name, body_names=morphology_configs.base_link_name_dict[asset_name]),
                    # Keep joints close to default standing pose with minimal initial motion
                    "position_range": (-0.3, 0.3),
                    "velocity_range": (-0.0, 0.0),
                }
            ))

            setattr(self, f"push_robot_{asset_name}", EventTerm(
                func=mdp.push_by_setting_velocity,
                mode="interval",
                interval_range_s=(3.0, 7.0),
                params={
                    "asset_cfg": SceneEntityCfg(asset_name, body_names=morphology_configs.base_link_name_dict[asset_name]),
                    "velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}
                },
            ))

@configclass
class IntegrationCurriculumCfg:
    pass
    # def __init__(self):
    #     import M2oE.configs.morphology_configs as morphology_configs
    #     for asset_name in morphology_configs.morphology_list:
    #         term_name_list = [ "is_alive", "diff_from_init_pose"]
    #         weight_list = [ 1.0, 5.0 ]
    #         for term_name, weight in zip(term_name_list, weight_list):
    #             setattr(self, term_name + f"_{asset_name}", CurriculumTermCfg(
    #                 func=mdp.modify_reward_weight_group,
    #                 params={
    #                     "group_name": f"reward_{asset_name}",
    #                     "term_name": term_name,
    #                     "weight": weight,
    #                     "num_steps": 8000, #20000 yue
    #                 },
    #             ))
                
    #     for asset_name in morphology_configs.morphology_list:
    #         # stage 2:  moving
    #         term_name_list = [ "track_lin_vel_xy_exp", "track_ang_vel_z_exp", "lin_vel_z_l2", "ang_vel_xy_l2", "wheel_ang_vel", "wheel_same_act" ]
    #         weight_list = [ 6.0, 3.0, -2.0, -0.1, 3.0, 3.0 ]
    #         for term_name, weight in zip(term_name_list, weight_list):
    #             setattr(self, term_name + f"_{asset_name}", CurriculumTermCfg(
    #                 func=mdp.modify_reward_weight_group,
    #                 params={
    #                     "group_name": f"reward_{asset_name}",
    #                     "term_name": "track_lin_vel_xy_exp",
    #                     "weight": weight,
    #                     "num_steps": 10000,
    #                 },
    #             ))
    #         # stage 3: polishing action
    #         term_name_list = [ "dof_acc_l2", "power", "dof_vel_l2", "action_rate" ]
    #         weight_list = [ -1e-5, -3e-4, -0.01, -0.0001 ]
    #         for term_name, weight in zip(term_name_list, weight_list):
    #             setattr(self, term_name + f"_{asset_name}", CurriculumTermCfg(
    #                 func=mdp.modify_reward_weight_group,
    #                 params={
    #                     "group_name": f"reward_{asset_name}",
    #                     "term_name": term_name,
    #                     "weight": weight,
    #                     "num_steps": 15000,
    #                 },
    #             ))

@configclass
class IntegrationViewerCfg(ViewerCfg):
    eye: tuple[float, float, float] = (7.5, 7.5, 7.5)
    lookat: tuple[float, float, float] = (0.0, 0.0, 0.0)
    cam_prim_path: str = "/OmniverseKit_Persp"
    resolution: tuple[int, int] = (1280, 720)
    origin_type: str = "asset_root" # type: ignore
    env_index: int = 0
    asset_name: str = "moonbot_full" # type: ignore

@configclass
class IntegrationEnvCfg(ManagerBasedRLEnvCfg):
    scene: IntegrationSceneCfg = IntegrationSceneCfg(num_envs=4096, env_spacing=5) # type: ignore

    observations: IntegrationObsCfg = IntegrationObsCfg()  # type: ignore
    actions: IntegrationActCfg = IntegrationActCfg()  # type: ignore
    commands: IntegrationCmdCfg = IntegrationCmdCfg()  # type: ignore

    rewards: IntegrationRewardCfg = IntegrationRewardCfg()  # type: ignore
    terminations: IntegrationTerminationCfg = IntegrationTerminationCfg() # type: ignore

    events: IntegrationEventCfg = IntegrationEventCfg() # type: ignore
    curriculum: IntegrationCurriculumCfg = IntegrationCurriculumCfg() # type: ignore
    viewer: IntegrationViewerCfg = IntegrationViewerCfg() # type: ignore

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0

        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2 ** 20
        self.sim.dt = 0.005
        self.sim.gravity = (0.0, 0.0, -1.62)
