"""Script to play a checkpoint from an M2oE agent."""

import argparse

from isaaclab.app import AppLauncher

# local imports
import M2oE.utils.cli_args as cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a trained RL agent with M2oE modules.")
parser.add_argument("--video", action="store_true", default=True, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=1000, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Integration_Locomotion_Play_v1", help="Name of the task.")
parser.add_argument("--real_time", action="store_true", default=False, help="Run in real-time, if possible.")

cli_args.add_m2oe_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True
args_cli.headless = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import os
import time
import torch

import moonbot_envs  # noqa: F401

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from M2oE.models.modules.on_policy_runner import OnPolicyRunner
from M2oE.utils.env_wrapper import ModulerRobotEnvWrapper
from M2oE.utils.utils import process_observations

from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg


def main():
    """Play with M2oE agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg = cli_args.parse_m2oe_cfg(args_cli.task, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("M2oE", "logs", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    env_cfg.scene.ground.terrain_generator.num_height = 1
    size_now = env_cfg.scene.ground.terrain_generator.size
    env_cfg.scene.ground.terrain_generator.size = (size_now[0], size_now[1], 0.0001)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        log_dir = os.path.dirname(resume_path)
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

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
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            actions = policy(obs, global_obs, module_masks)
            # env stepping
            obs, _, _, _ = env.step(actions.to(env.unwrapped.device))
            obs, global_obs, module_masks = process_observations(
                obs, num_obs=env.num_obs, num_global_obs=env.num_global_obs, policy=runner.alg.policy, num_envs=env.num_envs
            )
            obs, global_obs, module_masks = (
                obs.to(env.unwrapped.device),
                global_obs.to(env.unwrapped.device),
                module_masks.to(env.unwrapped.device),
            )
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
