from moonbot_envs.assets.moonbot import UNI_LEGGED_MOONBOT_CFG, DRAGON_MOONBOT_CFG, TRI_LEGGED_MOONBOT_CFG

morphology_list = [
    "moonbot_minimal",
    "moonbot_dragon",
    "moonbot_full",
]

morphology_asset_cfg = {
    "moonbot_minimal": UNI_LEGGED_MOONBOT_CFG,
    "moonbot_dragon": DRAGON_MOONBOT_CFG,
    "moonbot_full": TRI_LEGGED_MOONBOT_CFG,
}

morphology_asset_init_state = {
    "moonbot_minimal": {
        "base_position": [0.0, 0.0, 0.4],
        "base_orientation": [0.0, 0.0, 0.0, 1.0],
    },
    "moonbot_dragon": {
        "base_position": [0.0, 0.0, 0.48],
        "base_orientation": [0.0, 0.0, 0.0, 1.0],
    },
    "moonbot_full": {
        "base_position": [0.0, 0.0, 0.7],
        "base_orientation": [0.0, 0.0, 0.0, 1.0],
    },
}


wheel_link_name_dict = {
    "moonbot_minimal": [
        "Wheel_Link1",
        "Wheel_Link2",
    ],

    "moonbot_dragon": [
        "wheel14_left",
        "wheel14_right",
        "wheel12_left",
        "wheel12_right",
    ],
    "moonbot_full": [
        "leg1_wheel_left",
        "leg1_wheel_right",
        "leg2_wheel_left",
        "leg2_wheel_right",
        "leg3_wheel_left",
        "leg3_wheel_right",
    ],
}

base_link_name_dict = {
    "moonbot_minimal": "base_link",
    "moonbot_dragon": "base_link",
    "moonbot_full": "base_link",
}

contact_undesired_dict = {
    "moonbot_minimal": ["base_link", "Arm_Link1", "Arm_Link2", "Arm_Link3", "Arm_Link4", "Arm_Link5", "Arm_Link6", "Arm_Link7"],
    "moonbot_dragon": ["leg4link3", "leg4link4", "leg3link3", "leg3link4"],
    "moonbot_full": ["base_link"],
}

joint_names_dict = {
    "moonbot_minimal": [{
        "leg": ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"],
        "wheel": ["Wheel_joint1", "Wheel_joint2"],
    }],

    "moonbot_dragon": [{
        "leg": ["leg3joint1", "leg3joint2", "leg3joint3", "leg3joint4", "leg3joint5", "leg3joint6", "leg3joint7"],
        "wheel": ["wheel12_left_joint", "wheel12_right_joint"],
        # "leg": "leg3joint[1-7]",
        # "wheel": "wheel12_(left|right)_joint",
    },
    {
        "leg": ["leg4joint1", "leg4joint2", "leg4joint3", "leg4joint4", "leg4joint5", "leg4joint6", "leg4joint7"],
        "wheel": ["wheel14_left_joint", "wheel14_right_joint"],
        # "leg": "leg4joint[1-7]",
        # "wheel": "wheel14_(left|right)_joint",
    }],

    "moonbot_full": [{
        "leg": ["leg1_joint1", "leg1_joint2", "leg1_joint3", "leg1_joint4", "leg1_joint5", "leg1_joint6", "leg1_steering_joint"],
        "wheel": ["leg1_wheel_left_joint", "leg1_wheel_right_joint"],
        # "leg": "leg1_joint[1-6]|leg1_steering_joint",
        # "wheel": "leg1_wheel_(left|right)_joint",
    },
    {
        "leg": ["leg2_joint1", "leg2_joint2", "leg2_joint3", "leg2_joint4", "leg2_joint5", "leg2_joint6", "leg2_steering_joint"],
        "wheel": ["leg2_wheel_left_joint", "leg2_wheel_right_joint"],
        # "leg": "leg2_joint[1-6]|leg2_steering_joint",
        # "wheel": "leg2_wheel_(left|right)_joint",
    },
    {
        "leg": ["leg3_joint1", "leg3_joint2", "leg3_joint3", "leg3_joint4", "leg3_joint5", "leg3_joint6", "leg3_steering_joint"],
        "wheel": ["leg3_wheel_left_joint", "leg3_wheel_right_joint"],
        # "leg": "leg3_joint[1-6]|leg3_steering_joint",
        # "wheel": "leg3_wheel_(left|right)_joint",
    }]}
