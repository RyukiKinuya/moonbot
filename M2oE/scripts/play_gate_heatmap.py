"""Script to play a checkpoint and visualize MoE gate activations as a heatmap."""
import argparse

from isaaclab.app import AppLauncher

import M2oE.utils.cli_args as cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a trained RL agent and collect gate activations.")
parser.add_argument("--num_steps", type=int, default=2000, help="Number of steps to simulate.")
parser.add_argument("--heatmap_path", type=str, default="gate_heatmap.png", help="Path to save the figure.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during play.")
parser.add_argument("--video_length", type=int, default=1000, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=12, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Integration_Locomotion_v1", help="Name of the task.")
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

import math
import matplotlib
import os
import time

matplotlib.use("Agg")
import gymnasium as gym
import matplotlib.pyplot as plt
import torch

import moonbot_envs  # noqa: F401

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from M2oE.configs import morphology_configs
from M2oE.models.modules.on_policy_runner import OnPolicyRunner
from M2oE.utils.env_wrapper import ModulerRobotEnvWrapper
from M2oE.utils.utils import process_observations

from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg


def _compute_gate(model, obs, global_obs, module_masks):
    """Compute the gate activations of the M2oE model."""
    batch_size = obs.shape[0]
    max_num_modules = model.max_num_modules
    obs = obs.view(batch_size, max_num_modules, -1)
    if module_masks is not None:
        module_masks = module_masks.view(batch_size, max_num_modules)
    if model.gate_type == "linear":
        module_onehot = torch.eye(max_num_modules, device=obs.device).unsqueeze(0).expand(batch_size, -1, -1)
        expert_global_obs = global_obs.unsqueeze(1).expand(-1, max_num_modules, -1)
        gate_input = torch.cat((module_onehot, obs, expert_global_obs), dim=-1)
        gate = model.gate(gate_input)
    elif model.gate_type == "attention":
        gate = model.gate(obs, global_obs, module_masks)
    else:
        raise ValueError(f"Unknown gate type: {model.gate_type}.")
    return gate


def main():
    """Play with M2oE agent and collect gate activations."""
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg = cli_args.parse_m2oe_cfg(args_cli.task, args_cli)

    log_root_path = os.path.join("M2oE", "logs", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    env_cfg.scene.ground.terrain_generator.num_height = 1
    size_now = env_cfg.scene.ground.terrain_generator.size
    env_cfg.scene.ground.terrain_generator.size = (size_now[0], size_now[1], 0.0001)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

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

    env = ModulerRobotEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)

    policy = runner.alg.policy
    inference_policy = runner.get_inference_policy(device=env.unwrapped.device)

    dt = env.unwrapped.step_dt

    obs, _ = env.get_observations()
    obs, global_obs, module_masks = process_observations(
        obs, num_obs=env.num_obs, num_global_obs=env.num_global_obs, policy=runner.alg.policy, num_envs=env.num_envs
    )
    obs, global_obs, module_masks = (
        obs.to(env.unwrapped.device),
        global_obs.to(env.unwrapped.device),
        module_masks.to(env.unwrapped.device),
    )

    num_morphs = env.num_morphologies
    max_num_modules = policy.actor.max_num_modules  # type: ignore
    num_experts = policy.actor.num_experts  # type: ignore
    gate_sum = torch.zeros(
        env.base_num_envs, num_morphs, max_num_modules, num_experts, device=env.unwrapped.device
    )
    module_counts = torch.zeros(env.base_num_envs, num_morphs, max_num_modules, device=env.unwrapped.device)

    timestep = 0
    while simulation_app.is_running() and timestep < args_cli.num_steps:
        start_time = time.time()
        with torch.inference_mode():
            actions = inference_policy(obs, global_obs, module_masks)
            gate = _compute_gate(policy.actor, obs, global_obs, module_masks)   # gate: [batch_size, max_num_modules, num_experts]
            gate = gate.view(env.base_num_envs, num_morphs, max_num_modules, num_experts)   # reshape to [num_envs, num_morphs, max_num_modules, num_experts]
            mask = module_masks.view(env.base_num_envs, num_morphs, max_num_modules).float()
            gate_sum += gate * mask.unsqueeze(-1)
            module_counts += mask

            obs, _, _, _ = env.step(actions.to(env.unwrapped.device))
            obs, global_obs, module_masks = process_observations(
                obs,
                num_obs=env.num_obs,
                num_global_obs=env.num_global_obs,
                policy=runner.alg.policy,
                num_envs=env.num_envs,
            )
            obs, global_obs, module_masks = (
                obs.to(env.unwrapped.device),
                global_obs.to(env.unwrapped.device),
                module_masks.to(env.unwrapped.device),
            )
        if args_cli.video:
            if timestep == args_cli.video_length:
                break
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)
        timestep += 1

    env.close()

    gate_avg = gate_sum / module_counts.unsqueeze(-1).clamp(min=1)
    gate_avg = gate_avg.mean(dim=0).cpu().numpy()

    morphs = morphology_configs.morphology_list
    module_counts_list = [len(morphology_configs.joint_names_dict[morph]) for morph in morphs]
    num_experts = gate_avg.shape[-1]
    total_modules = sum(module_counts_list)
    cols = 3
    rows = math.ceil(total_modules / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = axes.flatten()

    plot_idx = 0
    for morph_idx, morph in enumerate(morphs):
        for module_idx in range(module_counts_list[morph_idx]):
            weights = gate_avg[morph_idx, module_idx]
            ax = axes[plot_idx]
            ax.bar(range(num_experts), weights)
            ax.set_title(f"{morph} module {module_idx + 1}")
            ax.set_xlabel("Expert")
            ax.set_ylabel("Average gate weight")
            ax.set_ylim(0, 1)
            plot_idx += 1

    for ax in axes[plot_idx:]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(args_cli.heatmap_path)


if __name__ == "__main__":
    main()
    simulation_app.close()
