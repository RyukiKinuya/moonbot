"""Play an M2oE agent and plot actual vs commanded linear speed and yaw rate."""


import argparse

from isaaclab.app import AppLauncher

import M2oE.utils.cli_args as cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a trained RL agent and record velocity tracking.")
parser.add_argument("--num_steps", type=int, default=None, help="Number of steps to simulate (defaults to one episode).")
parser.add_argument("--figure_path", type=str, default="velocity_tracking.png", help="Path to save the plot.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate (fixed to 1).")
parser.add_argument("--task", type=str, default="Integration_Locomotion_v1", help="Name of the task.")
parser.add_argument("--real_time", action="store_true", default=False, help="Run in real-time, if possible.")

cli_args.add_m2oe_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# ensure a single environment and run headless
args_cli.num_envs = 1
args_cli.headless = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import matplotlib
import numpy as np
import os
import time

matplotlib.use("Agg")
import gymnasium as gym
import matplotlib.pyplot as plt
import torch

import moonbot_envs  # noqa: F401

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from M2oE.configs import morphology_configs
from M2oE.models.modules.on_policy_runner import OnPolicyRunner
from M2oE.utils.env_wrapper import ModulerRobotEnvWrapper
from M2oE.utils.utils import process_observations

from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg


def main():
    """Run the policy and record velocity tracking."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg = cli_args.parse_m2oe_cfg(args_cli.task, args_cli)

    # specify directory for loading the experiment
    log_root_path = os.path.join("M2oE", "logs", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # flatten terrain to keep robots close to each other
    env_cfg.scene.ground.terrain_generator.num_height = 1
    size_now = env_cfg.scene.ground.terrain_generator.size
    env_cfg.scene.ground.terrain_generator.size = (size_now[0], size_now[1], 0.0001)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for M2oE
    env = ModulerRobotEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    dt = env.unwrapped.step_dt

    # reset environment
    obs, _ = env.get_observations()
    obs, global_obs, module_masks = process_observations(
        obs, num_obs=env.num_obs, num_global_obs=env.num_global_obs, policy=runner.alg.policy, num_envs=env.num_envs
    )
    obs, global_obs, module_masks = (
        obs.to(env.unwrapped.device),
        global_obs.to(env.unwrapped.device),
        module_masks.to(env.unwrapped.device),
    )

    morphs = morphology_configs.morphology_list
    vel_logs: dict[str, list[np.ndarray]] = {m: [] for m in morphs}
    cmd_logs: dict[str, list[np.ndarray]] = {m: [] for m in morphs}

    num_steps = args_cli.num_steps or env.max_episode_length
    for _ in range(num_steps):
        start_time = time.time()
        with torch.inference_mode():
            actions = policy(obs, global_obs, module_masks)
            obs, _, _, _ = env.step(actions.to(env.unwrapped.device))
            obs, global_obs, module_masks = process_observations(
                obs, num_obs=env.num_obs, num_global_obs=env.num_global_obs, policy=runner.alg.policy, num_envs=env.num_envs
            )
            obs, global_obs, module_masks = (
                obs.to(env.unwrapped.device),
                global_obs.to(env.unwrapped.device),
                module_masks.to(env.unwrapped.device),
            )
        for morph in morphs:
            asset = env.unwrapped.scene[morph]
            lin_vel = asset.data.root_lin_vel_b[0, :2]
            ang_vel = asset.data.root_ang_vel_b[0, 2]
            lin_speed = torch.linalg.norm(lin_vel).unsqueeze(0)
            vel_logs[morph].append(torch.cat([lin_speed, ang_vel.unsqueeze(0)]).cpu().numpy())
            cmd = env.unwrapped.command_manager.get_command(f"base_velocity_{morph}")[0]
            cmd_lin_speed = torch.linalg.norm(cmd[:2]).unsqueeze(0)
            cmd_logs[morph].append(torch.cat([cmd_lin_speed, cmd[2].unsqueeze(0)]).cpu().numpy())

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()

    # plotting
    timesteps = np.arange(len(next(iter(vel_logs.values()))))
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for ax, morph in zip(axes, morphs):
        vel = np.stack(vel_logs[morph])
        cmd = np.stack(cmd_logs[morph])

        ax.plot(timesteps, vel[:, 0], label="lin_speed")
        ax.plot(timesteps, cmd[:, 0], "--", label="cmd_lin_speed")
        ax.plot(timesteps, vel[:, 1], label="ang_z")
        ax.plot(timesteps, cmd[:, 1], "--", label="cmd_ang_z")

        ax.set_ylabel("Velocity")
        ax.set_title(morph.replace("moonbot_", ""))
        ax.legend(loc="upper right", fontsize="small")
    axes[-1].set_xlabel("Timestep")
    fig.tight_layout()
    fig.savefig(args_cli.figure_path)
    print(f"[INFO] Saved figure to {args_cli.figure_path}")


if __name__ == "__main__":
    main()
    simulation_app.close()
