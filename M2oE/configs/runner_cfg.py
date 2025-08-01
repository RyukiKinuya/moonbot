"""Configuration for the M2oE runner."""

from __future__ import annotations

from dataclasses import MISSING
from typing import Literal

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl.rl_cfg import (RslRlDistillationAlgorithmCfg, RslRlDistillationStudentTeacherCfg,
                                       RslRlPpoAlgorithmCfg)


@configclass
class M2oECfg:
    """Configuration of the M2oE model."""

    class_name: str = "M2oE"
    """Name of the model class. Default is ``M2oE``."""

    hidden_dim: int = MISSING
    """Hidden dimension of expert networks and critic."""

    num_experts: int = MISSING
    """Number of expert networks."""

    activation: str = MISSING
    """Activation function used in experts and critic."""

    gate_type: Literal["linear", "attention"] = "attention"
    """Type of gating network. Default is ``"attention"``."""

    gate_embedding_dim: int = 64
    """Embedding dimension for the attention gate."""

    gate_num_heads: int = 4
    """Number of attention heads in the gate."""

    gate_dropout: float = 0.1
    """Dropout probability for the attention gate."""


@configclass
class M2oEActorCriticCfg:
    """Configuration of the M2oE actor-critic policy."""

    class_name: str = "M2oEActorCritic"
    """Policy class name. Default is ``M2oEActorCritic``."""

    init_noise_std: float = MISSING
    """Initial standard deviation for action sampling."""

    noise_std_type: Literal["scalar", "log"] = "scalar"
    """Parameterization of the action noise. Default is ``"scalar"``."""

    padding_mode: str = "learnable"
    """Method for generating padding vectors. Default is ``"learnable"``."""

    padding_method: str = "concat"
    """Strategy for combining observation padding. Default is ``"concat"``."""


@configclass
class M2oEOnPolicyRunnerCfg:
    """Configuration of the runner for on-policy algorithms."""

    seed: int = 42
    """The seed for the experiment. Default is 42."""

    device: str = "cuda:0"
    """The device for the rl-agent. Default is cuda:0."""

    num_steps_per_env: int = MISSING
    """The number of steps per environment per update."""

    max_iterations: int = MISSING
    """The maximum number of iterations."""

    empirical_normalization: bool = MISSING
    """Whether to use empirical normalization."""

    max_num_modules: int = MISSING
    """Maximum number of modules in modular observations."""

    policy: M2oEActorCriticCfg | RslRlDistillationStudentTeacherCfg = MISSING
    """The policy configuration."""

    m2oe: M2oECfg = MISSING
    """The M2oE model configuration."""

    algorithm: RslRlPpoAlgorithmCfg | RslRlDistillationAlgorithmCfg = MISSING
    """The algorithm configuration."""

    clip_actions: float | None = None
    """The clipping value for actions. If ``None``, then no clipping is done.

    .. note::
        This clipping is performed inside the :class:`RslRlVecEnvWrapper` wrapper.
    """

    save_interval: int = MISSING
    """The number of iterations between saves."""

    experiment_name: str = MISSING
    """The experiment name."""

    run_name: str = ""
    """The run name. Default is empty string.

    The name of the run directory is typically the time-stamp at execution. If the run name is not empty,
    then it is appended to the run directory's name, i.e. the logging directory's name will become
    ``{time-stamp}_{run_name}``.
    """

    logger: Literal["tensorboard", "neptune", "wandb"] = "wandb"
    """The logger to use. Default is wandb."""

    neptune_project: str = "isaaclab"
    """The neptune project name. Default is "isaaclab"."""

    wandb_project: str = "isaaclab"
    """The wandb project name. Default is "isaaclab"."""

    resume: bool = False
    """Whether to resume. Default is False."""

    load_run: str = ".*"
    """The run directory to load. Default is ".*" (all).

    If regex expression, the latest (alphabetical order) matching run will be loaded.
    """

    load_checkpoint: str = "model_.*.pt"
    """The checkpoint file to load. Default is ``"model_.*.pt"`` (all).

    If regex expression, the latest (alphabetical order) matching file will be loaded.
    """
