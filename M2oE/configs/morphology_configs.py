morphology_list = [
    "moonbot_minimal",
    "moonbot_dragon",
    "moonbot_full",
]

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

joint_expr_dict = {
    "moonbot_minimal": [{
        "leg": "joint[1-7]",
        "wheel": "Wheel_joint[1-2]",
    }],

    "moonbot_dragon": [{
        "leg": "leg3joint[1-7]",
        "wheel": "Wheel12_{left|right}_joint",
    },
    {
        "leg": "leg4joint[1-7]",
        "wheel": "Wheel14_{left|right}_joint",
    }],

    "moonbot_full": [{
        "leg": "leg1joint[1-7]",
        "wheel": "leg1_wheel_{left|right}_joint",
    },
    {
        "leg": "leg2joint[1-7]",
        "wheel": "leg2_wheel_{left|right}_joint",
    },
    {
        "leg": "leg3joint[1-7]",
        "wheel": "leg3_wheel_{left|right}_joint",
    }]}