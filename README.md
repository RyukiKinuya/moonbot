# MoonBot RL Environments

This repository extends the Isaac Lab framework with a variety of lunar robot (MoonBot) tasks and provides training scripts as well as reinforcement learning implementations.

## Directory Structure
- `source/moonbot_envs`: definitions for environments and assets, task configuration, and the code that registers these environments with Gym.
- `M2oE/models`: reinforcement learning implementations based on RSL-RL, including modules such as `PPO`, `ActorCritic`, and `OnPolicyRunner`, along with the `ModulerRobotEnvWrapper` for a unified interface.
- `M2oE/scripts`: the training entry script `train.py`, which selects a task and starts training via command line arguments.

## Registered Environments
Environments are registered in `source/moonbot_envs/moonbot_envs/envs/__init__.py`. Example snippet:
```python
gym.register(
    id="Tri_Legged_Manipulate_v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": TriLeggedManipulateEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.MoonBotManipulatePPORunnerCfg
    },
)
```
Available tasks include learning environments for three-legged, single-legged, and dragon-style robots across scenarios such as grasping, locomotion, and terrain traversal.

## Training Script
`M2oE/scripts/train.py` provides a unified entry point for training. Example usage:
```bash
python M2oE/scripts/train.py --task Tri_Legged_Manipulate_v1 --num_envs 16
```
The script creates the corresponding Gym environment and uses `ModulerRobotEnvWrapper` to handle observation and action dimensions before training with `OnPolicyRunner`.

## Modular Robot Setup
Each task instantiates `num_morphology` modular robots inside every Isaac Lab environment. Training treats
all robots across environments as independent workers: with `num_env` simulator instances the policy sees
`num_env * num_morphology` parallel agents. Observations from different morphologies are padded to a common
size and actions are sliced back into groups before being passed to the simulator.
The wrapper assembles these groups into a dictionary keyed by the robot morphology
name so that the `GroupActionManager` can apply them correctly. The M2oE model
inside `M2oE/models` leverages the padded observations to output actions for all
modules simultaneously.

## Requirements and Usage
- First install [Isaac Lab](https://github.com/isaac-sim/IsaacLab) along with its dependencies.
- Using a GPU-enabled environment is recommended for the best simulation and training performance.

More details about environment configuration and algorithm specifics can be found in the code under each subdirectory.
