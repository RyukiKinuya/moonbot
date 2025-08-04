from isaaclab.utils import configclass
from M2oE.configs.runner_cfg import M2oEActorCriticCfg, M2oEOnPolicyRunnerCfg, MLPBaselineCfg

from isaaclab_rl.rsl_rl.rl_cfg import RslRlPpoAlgorithmCfg


@configclass
class JointMLP_Cfg(M2oEOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_num_modules = 3
    max_iterations = 2000
    save_interval = 100
    experiment_name = "MoonBot_locomotion"
    empirical_normalization = False
    policy = M2oEActorCriticCfg(
        init_noise_std=1.0,
    )
    model = MLPBaselineCfg(
        class_name="JointMLPBaseline",
        hidden_dims=(64, 64),
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.001,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
