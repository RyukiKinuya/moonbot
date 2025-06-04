# MoonBot RL Environments

本仓库在 Isaac Lab 框架基础上扩展了多种月球机器人（MoonBot）任务，并提供了训练脚本和强化学习实现。

## 目录结构
- `source/moonbot_envs`：环境与资产定义、任务配置以及在 Gym 中的注册代码。
- `M2oE/models`：基于 RSL-RL 的强化学习实现，包括 `PPO`、`ActorCritic`、`OnPolicyRunner` 等模块，以及用于统一接口的 `ModulerRobotEnvWrapper`。
- `M2oE/scripts`：训练入口脚本 `train.py`，通过命令行参数选择任务并启动训练。

## 已注册的环境
环境在 `source/moonbot_envs/moonbot_envs/envs/__init__.py` 中注册，示例片段如下：
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
可用任务包括三足、独腿以及龙形机器人在抓取、行走、地形穿越等多种场景下的学习环境。

## 训练脚本
`M2oE/scripts/train.py` 提供了统一的训练入口，使用示例：
```bash
python M2oE/scripts/train.py --task Tri_Legged_Manipulate_v1 --num_envs 16
```
脚本会创建对应的 Gym 环境，并通过 `ModulerRobotEnvWrapper` 处理观测与动作维度后交由 `OnPolicyRunner` 进行训练。

## 依赖与运行
- 需要先安装 [Isaac Lab](https://github.com/isaac-sim/IsaacLab) 及其依赖。
- 推荐使用带 GPU 的环境以获得最佳模拟和训练速度。

更多环境配置和算法细节可在各子目录的代码中找到。
