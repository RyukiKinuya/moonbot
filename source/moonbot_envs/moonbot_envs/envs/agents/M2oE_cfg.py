from isaaclab.utils import configclass
from M2oE.configs.runner_cfg import M2oEActorCriticCfg, M2oECfg, M2oEOnPolicyRunnerCfg

from isaaclab_rl.rsl_rl.rl_cfg import RslRlPpoAlgorithmCfg


@configclass
class M2oE_Cfg(M2oEOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_num_modules = 3
    max_iterations = 10000
    save_interval = 100
    experiment_name = "MoonBot_locomotion"
    empirical_normalization = False
    gate_warmup_steps = 100
    policy = M2oEActorCriticCfg(
        init_noise_std=1.0,
    )
    model = M2oECfg(
        hidden_dim=64,
        num_experts=6,
        activation="elu",
        gate_type="attention",
        gate_embedding_dim=32,
        gate_num_heads=4,
        gate_num_layers=1,
        gate_dropout=0.1,
        gate_use_positional_embedding=False,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.001,
        load_balance_loss_coef=0.1,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
