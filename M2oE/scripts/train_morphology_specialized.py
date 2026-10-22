"""Train three morphology-shared M2oE policies across nine streams.

Each morphology owns one M2oE. Its locomotion, arm-reach, and obstacle-avoid
environments are collected together and passed through task-specific adapters
before the shared policy. This is separate from ``M2oE/scripts/train.py``.
"""
from __future__ import annotations

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train three morphology-shared M2oE policies.")
parser.add_argument("--stream", action="append", required=True,
                    help="task=morphology=GymTaskId; repeat exactly nine times")
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--iterations", type=int, default=10000)
parser.add_argument("--device", type=str, default=None)
parser.add_argument("--seed", type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import moonbot_envs  # noqa: E402,F401
import moonbot_envs.envs.config.specialized_registration  # noqa: E402,F401
import torch  # noqa: E402
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402
from M2oE.models.morphology_specialized_moe import (  # noqa: E402
    MorphologySpecializedM2oE, ObservationSpec,
)
from M2oE.utils.env_wrapper import ModulerRobotEnvWrapper  # noqa: E402

TASKS = ("locomotion", "arm_reach", "obstacle_avoiding")
MORPHOLOGIES = ("minimal", "dragon", "tricycle")


def parse_streams(items):
    result = {}
    for item in items:
        task, morphology, task_id = item.split("=", 2)
        if task not in TASKS or morphology not in MORPHOLOGIES:
            raise ValueError(f"Invalid stream {item!r}")
        result[(task, morphology)] = task_id
    expected = {(task, morphology) for task in TASKS for morphology in MORPHOLOGIES}
    if set(result) != expected:
        raise ValueError(f"Exactly nine unique streams are required; missing={sorted(expected - set(result))}")
    return result


def main():
    streams = parse_streams(args_cli.stream)
    envs = {}
    specs = {}
    for (task, morphology), task_id in sorted(streams.items()):
        env_cfg = load_cfg_from_registry(task_id, "env_cfg_entry_point")
        env_cfg.scene.num_envs = args_cli.num_envs
        env_cfg.seed = args_cli.seed
        if args_cli.device:
            env_cfg.sim.device = args_cli.device
        env = gym.make(task_id, cfg=env_cfg, device=env_cfg.sim.device)
        if isinstance(env.unwrapped, DirectMARLEnv):
            env = multi_agent_to_single_agent(env)
        wrapped = ModulerRobotEnvWrapper(env, clip_actions=None)
        obs, extras = wrapped.get_observations()
        name = (task, morphology)
        specs[name] = ObservationSpec(
            num_obs=wrapped.num_obs,
            num_global_obs=wrapped.num_global_obs,
            num_actions=wrapped.num_actions,
            max_num_modules=wrapped.unwrapped.reward_manager.num_groups(),
        )
        envs[name] = wrapped

    # The policy and adapters are now dimension-checked from the actual Gym
    # environments, rather than from hand-written constants.
    try:
        agent_cfg = load_cfg_from_registry(next(iter(streams.values())), "M2oE_entry_point")
    except Exception:
        from moonbot_envs.envs.agents.M2oE_cfg import M2oE_Cfg
        agent_cfg = M2oE_Cfg()
    bank = MorphologySpecializedM2oE(
        specs, agent_cfg.model.to_dict(), agent_cfg.policy.to_dict(),
        device=agent_cfg.device,
    )
    optimizers = bank.optimizers(learning_rate=agent_cfg.algorithm.learning_rate)
    print(f"[INFO] Created {len(bank.policies)} shared morphology policies and {len(optimizers)} optimizers.")
    print("[INFO] The coordinator must collect each morphology's three streams and call adapt_observations before PPO updates.")
    print("[INFO] Existing OnPolicyRunner cannot perform this shared-policy multi-stream update automatically.")
    for env in envs.values():
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
