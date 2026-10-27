"""Shared Play and evaluation runtime for both M2oE allocation conditions.

``allocation=task`` selects one policy per logical task.  Each selected
integrated environment contains Minimal, Dragon and Tricycle.
``allocation=morphology`` selects one policy per morphology and applies its
task-specific observation adapter.  The public Play/evaluation entry points
all import this module so checkpoint loading, rollouts and visualizations stay
consistent.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch

TASKS = ("locomotion", "arm_reach", "obstacle_avoiding")
MORPHOLOGIES = ("minimal", "dragon", "tricycle")
TASK_GYM_IDS = {
    "locomotion": "Integration_Locomotion_v1",
    "arm_reach": "Integration_ArmReach_v1",
    "obstacle_avoiding": "Integration_ObstacleAvoid_v1",
}


def add_stream_args(parser, default_task: str, default_morphology: str, *, allocation: str = "morphology"):
    parser.set_defaults(allocation=allocation)
    parser.add_argument("--task", default=default_task, help="Registered Gym task id for the selected environment.")
    morphology_help = (
        "Morphology policy key." if allocation == "morphology" else
        "Deprecated compatibility option; task-specialized policies cover all morphologies."
    )
    parser.add_argument("--morphology", choices=MORPHOLOGIES, default=default_morphology, help=morphology_help)
    parser.add_argument("--task-kind", choices=TASKS, default="locomotion",
                        help="Logical task key used in the policy bank.")
    parser.add_argument("--checkpoint", required=True,
                        help="Specialized bank checkpoint (saved by the corresponding trainer).")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--disable_fabric", action="store_true")




def add_evaluation_args(parser, default_gym_task: str | None = None, *, default_task: str = "locomotion", allocation: str = "morphology"):
    """Arguments for one task/morphology evaluation under either allocation."""
    parser.set_defaults(allocation=allocation)
    parser.add_argument("--task", dest="task_kind", choices=TASKS, default=default_task,
                        help="Logical task: locomotion, arm_reach, or obstacle_avoiding.")
    parser.add_argument("--morphology", choices=MORPHOLOGIES, default="minimal",
                        help=("Morphology policy key." if allocation == "morphology" else
                              "Deprecated compatibility option; ignored for task-specialized policies."))
    parser.add_argument("--gym-task", dest="task", default=default_gym_task,
                        help="Registered integrated Isaac Lab Gym task id. Defaults from --task.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--disable_fabric", action="store_true")
    parser.add_argument("--device", default=None)

def make_env(args, app_cfg):
    import gymnasium as gym
    import moonbot_envs  # noqa: F401
    import moonbot_envs.envs.config.specialized_registration  # noqa: F401
    from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
    from M2oE.utils.env_wrapper import ModulerRobotEnvWrapper
    from isaaclab_tasks.utils import parse_env_cfg
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    if args.allocation == "task" and (args.task is None or (args.task == "Integration_Locomotion_v1" and args.task_kind != "locomotion")):
        args.task = TASK_GYM_IDS[args.task_kind]
    env_cfg = parse_env_cfg(args.task, device=args.device or getattr(app_cfg, "device", "cuda"),
                            num_envs=args.num_envs, use_fabric=not args.disable_fabric)
    env = gym.make(args.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    try:
        agent_cfg = load_cfg_from_registry(args.task, "M2oE_entry_point")
    except Exception:
        # Standard ManagerBasedRLEnv registrations use RSL-RL; specialized
        # play owns its M2oE config and therefore falls back to the new config.
        from moonbot_envs.envs.agents.M2oE_cfg import M2oE_Cfg
        agent_cfg = M2oE_Cfg()
    env = ModulerRobotEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    return env, agent_cfg


def _state_payload(checkpoint: str, device: str):
    payload = torch.load(checkpoint, map_location=device)
    if isinstance(payload, torch.nn.Module):
        return payload, None
    if not isinstance(payload, dict):
        raise TypeError(f"Unsupported specialized checkpoint type: {type(payload)!r}")
    bank = payload.get("bank") or payload.get("model")
    state = payload.get("state_dict") or payload.get("model_state_dict") or payload
    return bank, state


def _build_selected_policy(args, env, agent_cfg):
    """Load a complete bank when present, otherwise reconstruct one policy."""
    device = str(env.unwrapped.device)
    bank, state = _state_payload(args.checkpoint, device)
    key = args.task_kind if args.allocation == "task" else f"{args.task_kind}__{args.morphology}"
    if bank is not None:
        if args.allocation == "all_tasks":
            policy = bank.policy_for(args.task_kind, args.morphology) if hasattr(bank, "policy_for") else bank
        elif args.allocation == "task":
            try:
                policy = bank.policy_for(args.task_kind)
            except TypeError:
                policy = bank.policy_for(args.task_kind, args.morphology)
        else:
            policy = bank.policy_for(args.task_kind, args.morphology)
        return bank, policy
    spec = dict(num_obs=env.num_obs, num_global_obs=env.num_global_obs,
                num_actions=env.num_actions,
                max_num_modules=env.unwrapped.reward_manager.num_groups())
    from M2oE.models.modules.actor_critic import M2oEActorCritic
    policy = M2oEActorCritic(
        spec["num_obs"], spec["num_global_obs"], spec["num_actions"], spec["max_num_modules"],
        model_cfg=agent_cfg.model.to_dict(), device=device, **agent_cfg.policy.to_dict())
    prefix = f"policies.{key}."
    selected = {name[len(prefix):]: value for name, value in state.items()
                if isinstance(name, str) and name.startswith(prefix)}
    if not selected and args.allocation == "task":
        # Read checkpoints produced before the task-shared correction.
        legacy_key = f"{args.task_kind}__{args.morphology}"
        legacy_prefix = f"policies.{legacy_key}."
        selected = {name[len(legacy_prefix):]: value for name, value in state.items()
                    if isinstance(name, str) and name.startswith(legacy_prefix)}
    if not selected:
        selected = state.get(key, state)
    missing, unexpected = policy.load_state_dict(selected, strict=False)
    if missing:
        print(f"[WARN] Missing policy parameters for {key}: {len(missing)}")
    if unexpected:
        print(f"[WARN] Ignored checkpoint parameters for {key}: {len(unexpected)}")
    return None, policy


def open_stream(args, app_cfg):
    env, agent_cfg = make_env(args, app_cfg)
    bank, policy = _build_selected_policy(args, env, agent_cfg)
    policy.eval()
    return env, agent_cfg, bank, policy


def _prepare_obs(env, policy, raw_obs):
    from M2oE.utils.utils import process_observations
    obs, glob, masks = process_observations(
        raw_obs, num_obs=env.num_obs, num_global_obs=env.num_global_obs,
        policy=policy, num_envs=env.num_envs,
    )
    return obs.to(env.unwrapped.device), glob.to(env.unwrapped.device), masks.to(env.unwrapped.device)


def _policy_inputs(bank, args, obs, glob):
    if args.allocation == "morphology" and bank is not None and hasattr(bank, "adapt_observations"):
        policy, obs, glob = bank.adapt_observations(args.task_kind, args.morphology, obs, glob)
        return policy, obs, glob
    return None, obs, glob


def rollout(args, app_cfg, steps: int, callback=None):
    env, agent_cfg, bank, policy = open_stream(args, app_cfg)
    try:
        obs, _ = env.get_observations()
        obs, glob, masks = _prepare_obs(env, policy, obs)
        dt = getattr(env.unwrapped, "step_dt", 0.0)
        for step in range(steps):
            if hasattr(app_cfg, "is_running") and not app_cfg.is_running():
                break
            start = time.time()
            with torch.inference_mode():
                selected, pobs, pglob = _policy_inputs(bank, args, obs, glob)
                active = selected or policy
                actions = active.act_inference(pobs, pglob, masks)
                metrics = callback(step, active, pobs, pglob, masks, env) if callback else None
                obs, _, _, _ = env.step(actions.to(env.unwrapped.device))
                obs, glob, masks = _prepare_obs(env, policy, obs)
            if getattr(args, "real_time", False) and dt > 0:
                time.sleep(max(0.0, dt - (time.time() - start)))
            if metrics is not None:
                yield metrics
    finally:
        env.close()


def gate_values(policy, obs, glob, masks):
    actor = getattr(policy, "actor", policy)
    gate = getattr(actor, "gate", None)
    if gate is None:
        raise RuntimeError("The selected policy has no gate module.")
    b, modules = obs.shape[0], actor.max_num_modules
    modular = obs.reshape(b, modules, -1)
    if getattr(actor, "gate_type", None) == "linear":
        onehot = torch.eye(modules, device=obs.device).unsqueeze(0).expand(b, -1, -1)
        gobs = glob.unsqueeze(1).expand(-1, modules, -1)
        return gate(torch.cat((onehot, modular, gobs), dim=-1))
    return gate(modular, glob, masks.reshape(b, modules) if masks is not None else None)


def save_gate_heatmap(values, path: str, title: str):
    import matplotlib.pyplot as plt
    matrix = values
    while matrix.ndim > 2:
        matrix = matrix.mean(dim=0)
    matrix = matrix.detach().cpu().numpy()
    plt.figure(figsize=(max(6, matrix.shape[0] * 0.55), 4))
    plt.imshow(matrix, aspect="auto", cmap="YlGn")
    plt.colorbar(label="gate weight")
    plt.title(title)
    plt.xlabel("expert")
    plt.ylabel("module")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def extract_queries(policy, obs, glob, masks):
    actor = getattr(policy, "actor", policy)
    gate = getattr(actor, "gate", None)
    if gate is None or not hasattr(gate, "encoder"):
        raise RuntimeError("Query PCA requires an attention gate.")
    b, modules = obs.shape[0], actor.max_num_modules
    modular = obs.reshape(b, modules, -1)
    x_mod = gate.input_projection_modular(modular)
    x_glob = gate.input_projection_global(glob)
    x = torch.cat([x_glob.unsqueeze(1), x_mod], dim=1)
    if gate.use_positional_embedding:
        idx = torch.arange(modules + 1, device=obs.device)
        x = x + gate.posi(idx).unsqueeze(0)
    x = gate.pre_enc_ln(x)
    mask = None
    if masks is not None:
        m = masks.reshape(b, modules)
        mask = torch.cat([torch.zeros(b, 1, dtype=torch.bool, device=obs.device), ~m], dim=1)
    h = gate.encoder(x, src_key_padding_mask=mask)[:, 1:]
    return torch.nn.functional.normalize(h, dim=-1)


def save_query_pca(samples, path: str, title: str):
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    x = torch.cat(samples, dim=0).numpy()
    if x.shape[0] < 2:
        raise RuntimeError("Need at least two query samples for PCA.")
    z = PCA(n_components=2).fit_transform(x)
    plt.figure(figsize=(6, 5))
    plt.scatter(z[:, 0], z[:, 1], s=4, alpha=0.4)
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _task_metrics(env, morphology):
    """Return task metrics for one morphology without mixing task failures.

    Obstacle avoidance is a task-level result.  Its reported diagnostics are
    ``obstacle_avoidance_minimum_clearance`` and
    ``obstacle_avoidance_collision_rate``.  The latter is the fraction of
    environments whose contact force exceeds the configured threshold.
    """
    result = {
        "velocity_tracking_linear": None,
        "velocity_tracking_angular": None,
        "arm_reach_distance": None,
        "obstacle_avoidance_minimum_clearance": None,
        "obstacle_avoidance_collision_rate": None,
    }
    scene = getattr(env.unwrapped, "scene", None)
    commands = getattr(env.unwrapped, "command_manager", None)
    if scene is None or commands is None:
        return result
    asset_name = morphology if morphology in scene else {
        "minimal": "moonbot_minimal", "dragon": "moonbot_dragon", "tricycle": "moonbot_full",
    }.get(morphology, morphology)
    try:
        asset = scene[asset_name]
    except (KeyError, IndexError, AttributeError):
        return result

    # Locomotion diagnostic.  Reach and obstacle metrics remain available when
    # a task does not define velocity commands.
    try:
        velocity = commands.get_command(f"base_velocity_{asset_name}")
        result["velocity_tracking_linear"] = (
            asset.data.root_lin_vel_b[:, 0] - velocity[:, 0]
        ).abs().mean().item()
        result["velocity_tracking_angular"] = (
            asset.data.root_ang_vel_b[:, 2] - velocity[:, 2]
        ).abs().mean().item()
    except (KeyError, AttributeError, RuntimeError):
        pass

    ee = None
    try:
        ee_name = {"minimal": "Arm_Link7", "dragon": "gripper_palm", "tricycle": "gripper_palm"}[morphology]
        ee_id = asset.find_bodies(ee_name)[0][0]
        ee = asset.data.body_pos_w[:, ee_id] - env.unwrapped.scene.env_origins
        goal = commands.get_command(f"ee_pose_{asset_name}")[:, :3]
        result["arm_reach_distance"] = torch.linalg.vector_norm(goal - ee, dim=-1).mean().item()
    except (KeyError, IndexError, AttributeError, RuntimeError):
        pass

    # Obstacle avoidance diagnostics.  Clearance is a distance diagnostic;
    # collision rate is a contact diagnostic, not a separate task.
    if ee is not None:
        try:
            centers = torch.tensor(((0.65, -0.25, 0.95), (0.90, 0.0, 1.05),
                                    (0.65, 0.25, 0.95)), device=ee.device, dtype=ee.dtype)
            result["obstacle_avoidance_minimum_clearance"] = torch.linalg.vector_norm(
                centers.unsqueeze(0) - ee.unsqueeze(1), dim=-1
            ).amin(dim=-1).mean().item()
        except (AttributeError, RuntimeError):
            pass
    try:
        sensor_name = f"contact_forces_{asset_name}"
        if hasattr(scene, "sensors") and sensor_name in scene.sensors:
            force = scene.sensors[sensor_name].data.net_forces_w
            result["obstacle_avoidance_collision_rate"] = (
                torch.linalg.vector_norm(force, dim=-1).amax(dim=-1) > 12.0
            ).float().mean().item()
    except (KeyError, AttributeError, RuntimeError):
        pass
    return result


def run_all_evaluations(args, app, condition: str, heatmap_path: str, pca_path: str):
    """Run all five evaluations and write one gate image and one PCA image."""
    import collections
    gate_samples, query_samples = [], []
    values = collections.defaultdict(list)
    def callback(_, policy, obs, glob, masks, env):
        try:
            gate_samples.append(gate_values(policy, obs, glob, masks).detach().cpu())
        except RuntimeError:
            pass
        try:
            q = extract_queries(policy, obs, glob, masks)
            query_samples.append(q.reshape(-1, q.shape[-1]).detach().cpu())
        except RuntimeError:
            pass
        metrics = {}
        if args.allocation == "task":
            for morphology in MORPHOLOGIES:
                for metric, value in _task_metrics(env, morphology).items():
                    if value is not None:
                        metrics[f"{morphology}/{metric}"] = value
        else:
            metrics = _task_metrics(env, args.morphology)
        for key, value in metrics.items():
            if value is not None:
                values[key].append(value)
        return metrics

    for _ in rollout(args, app, args.num_steps, callback=callback):
        pass
    if gate_samples:
        save_gate_heatmap(torch.cat(gate_samples), heatmap_path,
                          (f"{condition} gate: {args.task_kind} (all morphologies)"
                           if args.allocation == "task" else
                           f"{condition} gate: {args.task_kind}/{args.morphology}"))
    if query_samples:
        save_query_pca([torch.cat(query_samples)], pca_path,
                       (f"{condition} key/query PCA: {args.task_kind} (all morphologies)"
                        if args.allocation == "task" else
                        f"{condition} key/query PCA: {args.task_kind}/{args.morphology}"))
    for key, samples in values.items():
        print(f"[RESULT] {condition} {key}: {sum(samples) / len(samples):.6f}")



def run_condition_evaluations(args, app, heatmap_path, pca_path):
    condition = "Task-specialized" if args.allocation == "task" else "Morphology-specialized"
    return run_all_evaluations(args, app, condition, heatmap_path, pca_path)
