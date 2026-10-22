"""Train the nine task/morphology streams with independent M2oE runners.

This entry point is separate from ``M2oE/scripts/train.py``. Each supplied
Gym task is wrapped and assigned its own OnPolicyRunner and optimizer. Use
``--stream task=morphology=GymTaskId`` once for every one of the nine streams.
The environment must expose ``M2oE_entry_point`` and use CustomManagerBasedRLEnv.
"""
from __future__ import annotations

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train nine independent task-specialized M2oE policies.")
parser.add_argument("--stream", action="append", required=True,
                    help="task=morphology=GymTaskId; repeat exactly nine times")
parser.add_argument("--num_envs", type=int, default=256)
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
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402
from M2oE.models.modules.on_policy_runner import OnPolicyRunner  # noqa: E402
from M2oE.utils.env_wrapper import ModulerRobotEnvWrapper  # noqa: E402

EXPECTED_TASKS = {"locomotion", "arm_reach", "obstacle_avoiding"}
EXPECTED_MORPHOLOGIES = {"minimal", "dragon", "tricycle"}


def parse_streams(items):
    streams = {}
    for item in items:
        fields = item.split("=", 2)
        if len(fields) != 3:
            raise ValueError(f"Invalid --stream {item!r}; expected task=morphology=GymTaskId")
        task, morphology, task_id = fields
        key = (task, morphology)
        if task not in EXPECTED_TASKS or morphology not in EXPECTED_MORPHOLOGIES:
            raise ValueError(f"Unknown stream key {key}; expected 3 tasks x 3 morphologies")
        if key in streams:
            raise ValueError(f"Duplicate stream {key}")
        streams[key] = task_id
    expected = {(task, morphology) for task in EXPECTED_TASKS for morphology in EXPECTED_MORPHOLOGIES}
    if set(streams) != expected:
        raise ValueError(f"Exactly nine streams are required; missing={sorted(expected - set(streams))}")
    return streams


def main():
    streams = parse_streams(args_cli.stream)
    runners = {}
    for (task, morphology), task_id in sorted(streams.items()):
        env_cfg = load_cfg_from_registry(task_id, "env_cfg_entry_point")
        try:
            agent_cfg = load_cfg_from_registry(task_id, "M2oE_entry_point")
        except Exception:
            from moonbot_envs.envs.agents.M2oE_cfg import M2oE_Cfg
            agent_cfg = M2oE_Cfg()
        env_cfg.scene.num_envs = args_cli.num_envs
        env_cfg.seed = args_cli.seed
        if args_cli.device:
            env_cfg.sim.device = args_cli.device
            agent_cfg.device = args_cli.device
        env = gym.make(task_id, cfg=env_cfg, device=env_cfg.sim.device)
        if isinstance(env.unwrapped, DirectMARLEnv):
            env = multi_agent_to_single_agent(env)
        wrapped = ModulerRobotEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        run_name = f"{task}__{morphology}"
        log_dir = os.path.abspath(os.path.join("M2oE", "logs", "TaskSpecialized", run_name))
        runners[(task, morphology)] = (OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=log_dir,
                                                       device=agent_cfg.device), wrapped)

    print(f"[INFO] Training {len(runners)} independent task-specialized M2oE runners.")
    try:
        # One update from every stream completes a round; this keeps all nine
        # policies advancing together while preserving independent optimizers.
        for round_idx in range(args_cli.iterations):
            for key in sorted(runners):
                runner, _ = runners[key]
                runner.learn(num_learning_iterations=1, init_at_random_ep_len=True)
            if (round_idx + 1) % 100 == 0:
                print(f"[INFO] completed synchronized round {round_idx + 1}/{args_cli.iterations}")
    finally:
        for _, env in runners.values():
            env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
