"""Train three task-specialized M2oE policies over the nine conditions.

Each ``--task-stream`` points to one integrated Isaac Lab environment.  The
integrated environment contains Minimal, Dragon and Tricycle simultaneously,
so one runner and one M2oE are shared by those three morphology modules.  Use
three entries, one for each logical task.
"""
from __future__ import annotations

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train three task-specialized M2oE policies.")
parser.add_argument(
    "--task-stream", "--stream", dest="task_stream", action="append", required=True,
    help="task=IntegratedGymTaskId; repeat exactly three times (one per task)",
)
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--iterations", type=int, default=10000)
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

EXPECTED_TASKS = ("locomotion", "arm_reach", "obstacle_avoiding")


def parse_task_streams(items):
    """Parse exactly one integrated Gym ID for each logical task."""
    streams = {}
    for item in items:
        fields = item.split("=", 1)
        if len(fields) != 2:
            raise ValueError(f"Invalid --task-stream {item!r}; expected task=IntegratedGymTaskId")
        task, task_id = fields
        if task not in EXPECTED_TASKS:
            raise ValueError(f"Unknown task {task!r}; expected {EXPECTED_TASKS}")
        if task in streams:
            raise ValueError(f"Duplicate task stream {task!r}")
        streams[task] = task_id
    if set(streams) != set(EXPECTED_TASKS):
        raise ValueError(
            "Exactly three integrated task streams are required; "
            f"missing={sorted(set(EXPECTED_TASKS) - set(streams))}"
        )
    return streams


def main():
    streams = parse_task_streams(args_cli.task_stream)
    runners = {}
    for task in EXPECTED_TASKS:
        task_id = streams[task]
        env_cfg = load_cfg_from_registry(task_id, "env_cfg_entry_point")
        agent_cfg = load_cfg_from_registry(task_id, "M2oE_entry_point")
        env_cfg.scene.num_envs = args_cli.num_envs
        env_cfg.seed = args_cli.seed
        if args_cli.device:
            env_cfg.sim.device = args_cli.device
            agent_cfg.device = args_cli.device
        env = gym.make(task_id, cfg=env_cfg, device=env_cfg.sim.device)
        if isinstance(env.unwrapped, DirectMARLEnv):
            env = multi_agent_to_single_agent(env)
        wrapped = ModulerRobotEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        log_dir = os.path.abspath(os.path.join("M2oE", "logs", "TaskSpecialized", task))
        runners[task] = (OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=log_dir,
                                        device=agent_cfg.device), wrapped)

    print("[INFO] Training 3 task-specialized M2oE policies.")
    print("[INFO] Each policy is shared by Minimal, Dragon and Tricycle inside its task environment.")
    try:
        for round_idx in range(args_cli.iterations):
            for task in EXPECTED_TASKS:
                runner, _ = runners[task]
                runner.learn(num_learning_iterations=1, init_at_random_ep_len=True)
            if (round_idx + 1) % 100 == 0:
                print(f"[INFO] completed synchronized round {round_idx + 1}/{args_cli.iterations}")
    finally:
        for _, env in runners.values():
            env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
