"""Script to play a checkpoint and visualize expert keys & query features via PCA."""
import argparse
import tqdm

from isaaclab.app import AppLauncher

import M2oE.utils.cli_args as cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a trained RL agent and visualize key-query PCA.")
parser.add_argument("--num_steps", type=int, default=3000, help="Number of steps to simulate.")
parser.add_argument("--save_path", type=str, default="key_query_pca.png", help="Path to save the figure.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=64, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Integration_Locomotion_v1", help="Name of the task.")
parser.add_argument("--max_query_samples", type=int, default=500, help="Max query samples per module to keep.")

cli_args.add_m2oe_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
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
from sklearn.manifold import TSNE

plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.titleweight"] = "bold"

import moonbot_envs  # noqa: F401

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from M2oE.configs import morphology_configs
from M2oE.models.modules.on_policy_runner import OnPolicyRunner
from M2oE.utils.env_wrapper import ModulerRobotEnvWrapper
from M2oE.utils.utils import process_observations

from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg


def _extract_queries(gate_module, obs, global_obs, module_masks):
    """Run the gate forward pass and return raw query vectors H (unnormalized)."""
    batch_size = obs.shape[0]
    max_num_modules = gate_module.max_num_modulars

    obs = obs.reshape(batch_size, max_num_modules, -1)
    if module_masks is not None:
        module_masks = module_masks.reshape(batch_size, max_num_modules)

    feature_modular = gate_module.input_projection_modular(obs)
    feature_global = gate_module.input_projection_global(global_obs)
    feature_integration = torch.cat([feature_global.unsqueeze(1), feature_modular], dim=1)

    if gate_module.use_positional_embedding:
        posi_indices = torch.arange(max_num_modules + 1, device=gate_module.device)
        feature_integration += gate_module.posi(posi_indices).unsqueeze(0)

    feature_integration = gate_module.pre_enc_ln(feature_integration)

    if module_masks is not None:
        key_padding_mask = torch.cat(
            [torch.zeros(module_masks.shape[0], 1, dtype=torch.bool, device=gate_module.device), ~module_masks], dim=1
        )
    else:
        key_padding_mask = None

    encoded = gate_module.encoder(feature_integration, src_key_padding_mask=key_padding_mask)
    H = encoded[:, 1:, :]
    Hn = torch.nn.functional.normalize(H, dim=-1)
    return Hn


def main():
    """Play with M2oE agent and collect query features for PCA visualization."""
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

    # Fix velocity commands: forward at 1.0 m/s, heading=0
    for asset_name in morphology_configs.morphology_list:
        cmd_cfg = getattr(env_cfg.commands, f"base_velocity_{asset_name}")
        cmd_cfg.ranges.lin_vel_x = (1.0, 1.0)
        cmd_cfg.ranges.lin_vel_y = (0.0, 0.0)
        cmd_cfg.ranges.ang_vel_z = (0.0, 0.0)
        cmd_cfg.ranges.heading = (0.0, 0.0)
        cmd_cfg.rel_standing_envs = 0.0

    env = gym.make(args_cli.task, cfg=env_cfg)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    env = ModulerRobotEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)

    runner.alg.policy.eval()
    policy = runner.alg.policy
    inference_policy = runner.get_inference_policy(device=env.unwrapped.device)

    # Get gate module (attention gate)
    gate_module = policy.actor.gate  # type: ignore

    num_morphs = env.num_morphologies
    max_num_modules = policy.actor.max_num_modules  # type: ignore
    morphs = morphology_configs.morphology_list
    module_counts_list = [len(morphology_configs.joint_names_dict[morph]) for morph in morphs]

    # Storage for query features: dict[(morph_idx, module_idx)] -> list of tensors
    query_store: dict[tuple[int, int], list[torch.Tensor]] = {}
    for morph_idx in range(num_morphs):
        for mod_idx in range(module_counts_list[morph_idx]):
            query_store[(morph_idx, mod_idx)] = []

    obs, _ = env.get_observations()
    obs, global_obs, module_masks = process_observations(
        obs, num_obs=env.num_obs, num_global_obs=env.num_global_obs, policy=runner.alg.policy, num_envs=env.num_envs
    )
    obs, global_obs, module_masks = (
        obs.to(env.unwrapped.device),
        global_obs.to(env.unwrapped.device),
        module_masks.to(env.unwrapped.device),
    )

    for _ in tqdm.tqdm(range(args_cli.num_steps), desc="Collecting queries"):
        if not simulation_app.is_running():
            break
        with torch.inference_mode():
            actions = inference_policy(obs, global_obs, module_masks)

            # Extract normalized query vectors
            Hn = _extract_queries(gate_module, obs, global_obs, module_masks)
            # Hn: [base_num_envs * num_morphs, max_num_modules, embedding_dim]
            Hn = Hn.reshape(env.base_num_envs, num_morphs, max_num_modules, -1)
            mask = module_masks.reshape(env.base_num_envs, num_morphs, max_num_modules)

            for morph_idx in range(num_morphs):
                for mod_idx in range(module_counts_list[morph_idx]):
                    valid = mask[:, morph_idx, mod_idx]  # [base_num_envs]
                    valid_queries = Hn[valid, morph_idx, mod_idx]  # [N_valid, embedding_dim]
                    if valid_queries.shape[0] > 0:
                        query_store[(morph_idx, mod_idx)].append(valid_queries.cpu())

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

    env.close()

    # --- Gather data ---
    # Subsample queries per module
    query_arrays: dict[tuple[int, int], np.ndarray] = {}
    for key, tensors in query_store.items():
        if len(tensors) == 0:
            continue
        cat = torch.cat(tensors, dim=0).numpy()
        if cat.shape[0] > args_cli.max_query_samples:
            indices = np.random.choice(cat.shape[0], args_cli.max_query_samples, replace=False)
            cat = cat[indices]
        query_arrays[key] = cat

    # Stack all queries for t-SNE
    all_queries = np.concatenate([query_arrays[k] for k in sorted(query_arrays.keys())], axis=0)
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init="pca", learning_rate="auto")
    projected = tsne.fit_transform(all_queries)

    # Split back
    offset = 0
    query_proj: dict[tuple[int, int], np.ndarray] = {}
    for key in sorted(query_arrays.keys()):
        n = query_arrays[key].shape[0]
        query_proj[key] = projected[offset : offset + n]
        offset += n

    # --- Plot ---
    morph_short_names = ["Minimal", "Dragon", "Full"]
    morph_colors = ["#4E79A7", "#E15759", "#59A14F"]  # blue, red, green
    module_markers = ["o", "s", "D", "^", "v", "P"]

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.set_facecolor("#FAFAFA")

    # Scatter query points
    for (morph_idx, mod_idx), pts in query_proj.items():
        num_suffix = ["st", "nd", "rd"] + ["th"] * 7
        label = f"{morph_short_names[morph_idx]} {mod_idx+1}{num_suffix[mod_idx]} module"
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            c=morph_colors[morph_idx],
            marker=module_markers[mod_idx % len(module_markers)],
            s=15,
            alpha=0.5,
            label=label,
            rasterized=True,
            edgecolors="none",
        )

    # Build legend
    from matplotlib.lines import Line2D
    legend_handles = []
    for (morph_idx, mod_idx) in sorted(query_proj.keys()):
        num_suffix = ["st", "nd", "rd"] + ["th"] * 7
        legend_handles.append(Line2D(
            [0], [0],
            marker=module_markers[mod_idx % len(module_markers)],
            color="none",
            markerfacecolor=morph_colors[morph_idx],
            markeredgecolor="black",
            markersize=8,
            label=f"{morph_short_names[morph_idx]} {mod_idx+1}{num_suffix[mod_idx]} module",
        ))

    legend = ax.legend(
        handles=legend_handles,
        loc="best",
        fontsize=9,
        framealpha=0.9,
        edgecolor="gray",
    )
    for text in legend.get_texts():
        text.set_fontweight("bold")

    ax.set_xlabel("t-SNE 1", fontsize=13, fontweight="bold")
    ax.set_ylabel("t-SNE 2", fontsize=13, fontweight="bold")
    ax.set_title("Query Features (t-SNE)", fontsize=14, fontweight="bold")

    ax.tick_params(labelsize=11)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")

    ax.grid(True, alpha=0.2, linestyle="--")
    fig.tight_layout()
    plt.savefig(args_cli.save_path, dpi=600, bbox_inches="tight")
    print(f"[INFO] Saved PCA plot to {args_cli.save_path}")


if __name__ == "__main__":
    main()
    simulation_app.close()
