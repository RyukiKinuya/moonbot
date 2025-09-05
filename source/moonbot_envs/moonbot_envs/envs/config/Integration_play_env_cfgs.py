from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp import events as isaac_mdp_events
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.envs.common import ViewerCfg

from M2oE.configs import morphology_configs
import moonbot_envs.envs.mdp as mdp
from moonbot_envs.assets import *  # noqa: F401,F403
from moonbot_envs.custom_lab_envs.manager_term_cfg import ActionGroupCfg, RewardGroupCfg
from moonbot_envs.custom_lab_envs.terrains.config.wave_terrains_cfg import WAVE_TERRAINS_CFG
from moonbot_envs.custom_lab_envs.terrains.terrain_importer_cfg import TerrainImporterCfg


@configclass
class PlayIntegrationSceneCfg(InteractiveSceneCfg):
    def __init__(self, num_envs: int = 4096, env_spacing: float = 10.0):
        super().__init__(num_envs=num_envs, env_spacing=env_spacing)

        # Ground and lighting
        setattr(
            self,
            "ground",
            TerrainImporterCfg(
                prim_path="/World/ground",
                terrain_type="generator",
                terrain_generator=WAVE_TERRAINS_CFG,
                max_init_terrain_level=1,
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
            ),
        )

        setattr(
            self,
            "light",
            AssetBaseCfg(
                prim_path="/World/skyLight",
                spawn=sim_utils.DomeLightCfg(
                    intensity=750.0,
                    texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
                ),
            ),
        )

        # Assets and sensors
        for asset_name in morphology_configs.morphology_list:
            setattr(
                self,
                asset_name,
                morphology_configs.morphology_asset_cfg[asset_name]
                .replace(prim_path="{ENV_REGEX_NS}/" + asset_name)
                .replace(
                    init_state=ArticulationCfg.InitialStateCfg(
                        pos=morphology_configs.morphology_asset_init_state[asset_name]["base_position"],
                        joint_pos=morphology_configs.morphology_asset_init_state[asset_name]["joint_pos"],
                    )
                ),
            )

            setattr(
                self,
                f"contact_forces_{asset_name}",
                ContactSensorCfg(
                    prim_path="{ENV_REGEX_NS}/" + asset_name + "/" + morphology_configs.contact_sensor_links[asset_name],
                    history_length=1,
                    track_air_time=True,
                    force_threshold=8.0,
                ),
            )


@configclass
class PlayIntegrationObsCfg:
    @configclass
    class MoonbotObsCfg(ObsGroup):
        def __init__(self, asset_cfg: SceneEntityCfg, num_morphologies: int):
            super().__init__()
            for i in range(num_morphologies):
                module_name = f"module_{i}"
                setattr(
                    self,
                    module_name,
                    ObsTerm(func=mdp.module_obs, params={"asset_cfg": asset_cfg, "module_no": i}),
                )

        def __post_init__(self):
            # Disable observation corruption for play
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class GlobalCfg(ObsGroup):
        def __init__(self, asset_cfg: SceneEntityCfg = SceneEntityCfg("moonbot_minimal")):
            super().__init__()
            self.base_lin_vel = ObsTerm(func=mdp.base_lin_vel, params={"asset_cfg": asset_cfg})
            self.base_ang_vel = ObsTerm(func=mdp.base_ang_vel, params={"asset_cfg": asset_cfg})
            self.velocity_commands = ObsTerm(
                func=mdp.generated_commands,
                params={"command_name": f"base_velocity_{asset_cfg.name}"},
            )
            self.projected_gravity = ObsTerm(func=mdp.projected_gravity, params={"asset_cfg": asset_cfg})

        def __post_init__(self):
            # Disable observation corruption for play
            self.enable_corruption = False
            self.concatenate_terms = True

    obs_minimal: MoonbotObsCfg = MoonbotObsCfg(asset_cfg=SceneEntityCfg("moonbot_minimal"), num_morphologies=1)
    obs_dragon: MoonbotObsCfg = MoonbotObsCfg(asset_cfg=SceneEntityCfg("moonbot_dragon"), num_morphologies=2)
    obs_full: MoonbotObsCfg = MoonbotObsCfg(asset_cfg=SceneEntityCfg("moonbot_full"), num_morphologies=3)
    global_obs_moonbot_minimal: GlobalCfg = GlobalCfg(asset_cfg=SceneEntityCfg("moonbot_minimal"))
    global_obs_moonbot_dragon: GlobalCfg = GlobalCfg(asset_cfg=SceneEntityCfg("moonbot_dragon"))
    global_obs_moonbot_full: GlobalCfg = GlobalCfg(asset_cfg=SceneEntityCfg("moonbot_full"))


@configclass
class PlayIntegrationActCfg:
    @configclass
    class MoonbotActCfg(ActionGroupCfg):
        def __init__(self, asset_name: str, num_morphologies: int = 1):
            super().__init__()
            for i in range(num_morphologies):
                # leg actions
                setattr(
                    self,
                    f"module_{i}_action_leg",
                    mdp.JointPositionActionCfg(
                        asset_name=asset_name,
                        joint_names=morphology_configs.joint_names_dict[asset_name][i]["leg"],
                        scale=0.1,
                        use_default_offset=True,
                        preserve_order=True,
                    ),
                )
                # wheel actions
                setattr(
                    self,
                    f"module_{i}_action_wheel",
                    mdp.JointVelocityActionCfg(
                        asset_name=asset_name,
                        joint_names=morphology_configs.joint_names_dict[asset_name][i]["wheel"],
                        scale=10,
                        preserve_order=True,
                    ),
                )

    act_moonbot_minimal: MoonbotActCfg = MoonbotActCfg(asset_name="moonbot_minimal", num_morphologies=1)
    act_moonbot_dragon: MoonbotActCfg = MoonbotActCfg(asset_name="moonbot_dragon", num_morphologies=2)
    act_moonbot_full: MoonbotActCfg = MoonbotActCfg(asset_name="moonbot_full", num_morphologies=3)


@configclass
class PlayIntegrationCmdCfg:
    def __init__(self):
        # Only sample the velocity command once at reset (very large resampling time).
        long_t = 1e9
        for asset_name in morphology_configs.morphology_list:
            setattr(
                self,
                f"base_velocity_{asset_name}",
                mdp.UniformVelocityCommandCfg(
                    asset_name=asset_name,
                    resampling_time_range=(long_t, long_t),
                    rel_standing_envs=0.0,
                    rel_heading_envs=1.0,
                    heading_command=True,
                    heading_control_stiffness=0.5,
                    debug_vis=True,
                    ranges=mdp.UniformVelocityCommandCfg.Ranges(
                        lin_vel_x=(-1.2, 1.2),
                        lin_vel_y=(-0.0, 0.0),
                        ang_vel_z=(-math.pi / 3, math.pi / 3),
                        heading=(-math.pi, math.pi),
                    ),
                ),
            )


@configclass
class PlayIntegrationRewardCfg:
    @configclass
    class MoonbotRewardCfg(RewardGroupCfg):
        def __init__(self, asset_cfg: SceneEntityCfg):
            self.track_lin_vel_xy_exp = RewTerm(
                func=mdp.track_lin_vel_xy_exp,
                weight=6.0,
                params={"asset_cfg": asset_cfg, "command_name": f"base_velocity_{asset_cfg.name}", "std": math.sqrt(0.25)},
            )
            self.track_ang_vel_z_exp = RewTerm(
                func=mdp.track_ang_vel_z_exp,
                weight=3.0,
                params={"asset_cfg": asset_cfg, "command_name": f"base_velocity_{asset_cfg.name}", "std": math.sqrt(0.25)},
            )
            self.is_alive = RewTerm(func=mdp.is_alive, weight=0.0)
            self.diff_from_init_pose = RewTerm(func=mdp.diff_from_init_pose, weight=5.0, params={"asset_cfg": asset_cfg})
            self.wheel_rolling_resistance = RewTerm(
                func=mdp.wheel_rolling_consistency,
                weight=-1.0,
                params={"asset_cfg": asset_cfg, "command_name": f"base_velocity_{asset_cfg.name}"},
            )
            if asset_cfg.name == "moonbot_full":
                self.wheel_distance = RewTerm(func=mdp.wheel_distances, weight=1.0, params={"asset_cfg": asset_cfg})
            self.base_balance = RewTerm(func=mdp.base_balance, weight=-5.0, params={"asset_cfg": asset_cfg})
            self.undesired_contacts = RewTerm(
                func=mdp.undesired_contacts_moonbot,
                weight=-0.5,
                params={"sensor_cfg": SceneEntityCfg(f"contact_forces_{asset_cfg.name}"), "threshold": 1.0},
            )
            self.lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0, params={"asset_cfg": asset_cfg})
            self.ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.1, params={"asset_cfg": asset_cfg})
            self.dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2e-5, params={"asset_cfg": asset_cfg})
            self.power = RewTerm(func=mdp.joint_power, weight=-5e-4, params={"asset_cfg": asset_cfg})
            self.dof_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=-0.02, params={"asset_cfg": asset_cfg})
            self.action_rate = RewTerm(func=mdp.action_rate_modular, weight=-0.01, params={"asset_cfg": asset_cfg})

    reward_moonbot_minimal: MoonbotRewardCfg = MoonbotRewardCfg(
        asset_cfg=SceneEntityCfg("moonbot_minimal", joint_names=morphology_configs.joint_names_dict_flat["moonbot_minimal"]["leg"])
    )
    reward_moonbot_dragon: MoonbotRewardCfg = MoonbotRewardCfg(
        asset_cfg=SceneEntityCfg("moonbot_dragon", joint_names=morphology_configs.joint_names_dict_flat["moonbot_dragon"]["leg"])
    )
    reward_moonbot_full: MoonbotRewardCfg = MoonbotRewardCfg(
        asset_cfg=SceneEntityCfg("moonbot_full", joint_names=morphology_configs.joint_names_dict_flat["moonbot_full"]["leg"])
    )


@configclass
class PlayIntegrationTerminationCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact_minimal = DoneTerm(
        func=mdp.illegal_contact_moonbot,
        params={"sensor_cfg": SceneEntityCfg("contact_forces_moonbot_minimal", body_names="base_link"), "threshold": 1.0},
    )
    base_contact_dragon = DoneTerm(
        func=mdp.illegal_contact_moonbot,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces_moonbot_dragon",
                body_names="leg4link[3-4]|leg3link[3-6]|leg3gripper2|leg3gripper2_straight",
            ),
            "threshold": 1.0,
        },
    )
    base_contact_full = DoneTerm(
        func=mdp.illegal_contact_moonbot,
        params={"sensor_cfg": SceneEntityCfg("contact_forces_moonbot_full", body_names="base"), "threshold": 1.0},
    )


@configclass
class PlayIntegrationEventCfg:
    def __init__(self):
        # Deterministic reset of base pose/velocities (no random events during play)
        for asset_name in morphology_configs.morphology_list:
            # Reset roots to default poses (no randomness)
            setattr(
                self,
                f"reset_base_{asset_name}",
                EventTerm(
                    func=isaac_mdp_events.reset_root_state_uniform,
                    mode="reset",
                    params={
                        "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0), "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0)},
                        "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0), "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0)},
                        "asset_cfg": SceneEntityCfg(asset_name),
                    },
                ),
            )

            # Reset joints to default without randomness
            setattr(
                self,
                f"reset_joints_{asset_name}",
                EventTerm(
                    func=mdp.reset_joints_by_offset,
                    mode="reset",
                    params={
                        "asset_cfg": SceneEntityCfg(asset_name, body_names=morphology_configs.base_link_name_dict[asset_name]),
                        "position_range": (0.0, 0.0),
                        "velocity_range": (0.0, 0.0),
                    },
                ),
            )


@configclass
class PlayIntegrationViewerCfg(ViewerCfg):
    eye: tuple[float, float, float] = (7.5, 7.5, 7.5)
    lookat: tuple[float, float, float] = (0.0, 0.0, 0.0)
    cam_prim_path: str = "/OmniverseKit_Persp"
    resolution: tuple[int, int] = (1280, 720)
    origin_type: str = "asset_root"  # type: ignore
    env_index: int = 0
    asset_name: str = "moonbot_full"  # type: ignore


@configclass
class PlayIntegrationEnvCfg(ManagerBasedRLEnvCfg):
    scene: PlayIntegrationSceneCfg = PlayIntegrationSceneCfg(num_envs=4096, env_spacing=5)  # type: ignore

    observations: PlayIntegrationObsCfg = PlayIntegrationObsCfg()  # type: ignore
    actions: PlayIntegrationActCfg = PlayIntegrationActCfg()  # type: ignore
    commands: PlayIntegrationCmdCfg = PlayIntegrationCmdCfg()  # type: ignore

    rewards: PlayIntegrationRewardCfg = PlayIntegrationRewardCfg()  # type: ignore
    terminations: PlayIntegrationTerminationCfg = PlayIntegrationTerminationCfg()  # type: ignore

    events: PlayIntegrationEventCfg = PlayIntegrationEventCfg()  # type: ignore
    viewer: PlayIntegrationViewerCfg = PlayIntegrationViewerCfg()  # type: ignore

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0

        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2 ** 20
        self.sim.dt = 0.005
        self.sim.gravity = (0.0, 0.0, -1.62)
