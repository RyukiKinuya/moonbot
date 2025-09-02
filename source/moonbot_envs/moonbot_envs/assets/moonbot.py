from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg


TRI_LEGGED_MOONBOT_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path="usd/tricycle/tricycle.usda",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            max_linear_velocity=10.0,
            max_angular_velocity=31.4,
            max_contact_impulse=10,
            solver_position_iteration_count=10,
            solver_velocity_iteration_count=4,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
        copy_from_source=False,
        activate_contact_sensors=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.6),
        joint_pos={
            "leg1_joint7": -1.0471975,
            "leg2_joint7": 1.0471975,
        },
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["arm.*"],
            stiffness=10000000000.0,
            damping=80000000.0,
            effort_limit=87.0,
            velocity_limit=1.0,
        ),

        "leg": ImplicitActuatorCfg(
            joint_names_expr=["leg[1-3]_joint.*"],
            stiffness=800.0,
            damping=20.0,
            effort_limit=87.0,
            velocity_limit=1.0,
        ),

        "wheel": ImplicitActuatorCfg(
            joint_names_expr=[".*wheel.*"],
            stiffness=0.0,
            damping=100.0,
            velocity_limit=5.0,
        ),
    },
)

UNI_LEGGED_MOONBOT_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path="usd/moonbothm_urdf_v8.usd",
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True,
            contact_offset=0.1,
            rest_offset=0.01,
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.5,
            angular_damping=0.5,
            max_linear_velocity=1.0,
            max_angular_velocity=3.14,
            max_depenetration_velocity=0.5,
            max_contact_impulse=10,
            solver_position_iteration_count=10,
            solver_velocity_iteration_count=4,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=2,
            solver_velocity_iteration_count=1,
            fix_root_link=False,
        ),
        copy_from_source=False,
        activate_contact_sensors=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.4),
        rot=(0.707, 0.707, 0.0, 0.0),
        # rot=(0.0, 0.0, 0.0, 1.0),
        joint_pos={
            ".*": 0.0,
        },
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["joint.*"],
            stiffness=800.0,
            damping=20.0,
            effort_limit=87.0,
            velocity_limit=1.0,
        ),

        "wheel": ImplicitActuatorCfg(
            joint_names_expr=["Wheel.*"],
            stiffness=0.0,
            damping=100.0,
            velocity_limit=5.0,
        ),
    },
)


UNI_LEGGED_FIX_BASE_MOONBOT_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UrdfFileCfg(
        asset_path="usd/moonbothm_urdf_v8/moonbothm_urdf_v8/urdf/moonbothm_urdf_v8.urdf",
        fix_base=False,
        merge_fixed_joints=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=True,
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=100.0,
            max_angular_velocity=100.0,
            max_depenetration_velocity=0.5,
            max_contact_impulse=10,
            solver_position_iteration_count=10,
            solver_velocity_iteration_count=2,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=10,
            solver_velocity_iteration_count=2,
            fix_root_link=True,
        ),
        copy_from_source=False,
        activate_contact_sensors=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.4),
        rot=(0.707, 0.707, 0.0, 0.0),
        # rot=(0.0, 0.0, 0.0, 1.0),
        joint_pos={
            ".*": 0.0,
        },
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["joint.*"],
            stiffness=800.0,
            damping=20.0,
            effort_limit=87.0,
        ),

        "wheel": ImplicitActuatorCfg(
            joint_names_expr=["Wheel.*"],
            stiffness=200.0,
            damping=20.0,
        ),
    },
)

DRAGON_MOONBOT_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path="usd/hero_dragon.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=10.0,
            max_angular_velocity=10.0,
            max_depenetration_velocity=0.5,
            max_contact_impulse=10,
            solver_position_iteration_count=10,
            solver_velocity_iteration_count=2,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=2, solver_velocity_iteration_count=1
        ),
        copy_from_source=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5),
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["leg3joint.*"],
            stiffness=800.0,
            damping=20.0,
            effort_limit=87.0,
            velocity_limit=1.0,
        ),
        "connect_link_free": ImplicitActuatorCfg(
            joint_names_expr=["leg4joint[1-7]"],
            stiffness=800.0,
            damping=20.0,
            effort_limit=87.0,
            velocity_limit=1.0,
        ),
        # "connect_link": ImplicitActuatorCfg(
        #     joint_names_expr=["leg4joint[4-6]"],
        #     stiffness=1000000000.0,
        #     damping=4000000.0,
        #     effort_limit=87.0,
        # ),
        "wheel": ImplicitActuatorCfg(
            joint_names_expr=["wheel.*joint"],
            stiffness=0.0,
            damping=100.0,
            velocity_limit=5.0
        ),
    }
)
