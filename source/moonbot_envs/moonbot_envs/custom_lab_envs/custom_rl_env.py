from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.common import VecEnvStepReturn
from isaaclab.managers import CommandManager, CurriculumManager, ObservationManager, RecorderManager, TerminationManager

from .action_manager import GroupActionManager
from .reward_manager import RewardManager


class CustomManagerBasedRLEnv(ManagerBasedRLEnv):
    """Manager-based RL environment using the custom reward manager."""

    def load_managers(self):
        """Load and initialize environment managers with grouped action manager."""
        # Command manager first so observation manager can access commands
        self.command_manager: CommandManager = CommandManager(self.cfg.commands, self)
        print("[INFO] Command Manager: ", self.command_manager)

        # Managers from base environment (event manager already exists)
        print("[INFO] Event Manager: ", self.event_manager)
        self.recorder_manager = RecorderManager(self.cfg.recorders, self)
        print("[INFO] Recorder Manager: ", self.recorder_manager)

        # Group action manager before observation manager
        self.action_manager = GroupActionManager(self.cfg.actions, self)
        print("[INFO] Action Manager: ", self.action_manager)

        self.observation_manager = ObservationManager(self.cfg.observations, self)
        print("[INFO] Observation Manager:", self.observation_manager)

        # Termination, reward and curriculum managers
        self.termination_manager = TerminationManager(self.cfg.terminations, self)
        print("[INFO] Termination Manager: ", self.termination_manager)
        self.reward_manager = RewardManager(self.cfg.rewards, self)
        print("[INFO] Reward Manager: ", self.reward_manager)
        self.curriculum_manager = CurriculumManager(self.cfg.curriculum, self)
        print("[INFO] Curriculum Manager: ", self.curriculum_manager)

        # Setup Gym spaces
        self._configure_gym_env_spaces()

        # Trigger startup events if available
        if "startup" in self.event_manager.available_modes:
            self.event_manager.apply(mode="startup")

    def step(self, action: dict[str, torch.Tensor]) -> VecEnvStepReturn:
        """Step the environment with grouped actions."""
        # process grouped actions without sending to device
        self.action_manager.process_action(action)

        self.recorder_manager.record_pre_step()

        # check if we need to do rendering within the physics loop
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            self.action_manager.apply_action()
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            self.scene.update(dt=self.physics_dt)

        self.episode_length_buf += 1
        self.common_step_counter += 1
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)

        if len(self.recorder_manager.active_terms) > 0:
            self.obs_buf = self.observation_manager.compute()
            self.recorder_manager.record_post_step()

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.recorder_manager.record_pre_reset(reset_env_ids)
            self._reset_idx(reset_env_ids)
            self.scene.write_data_to_sim()
            self.sim.forward()
            if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
                self.sim.render()
            self.recorder_manager.record_post_reset(reset_env_ids)

        self.command_manager.compute(dt=self.step_dt)
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)

        self.obs_buf = self.observation_manager.compute()

        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras
