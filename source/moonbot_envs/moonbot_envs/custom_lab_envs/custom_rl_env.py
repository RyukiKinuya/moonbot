from __future__ import annotations

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import CommandManager, CurriculumManager, TerminationManager, RecorderManager, EventManager
from isaaclab.managers import ObservationManager

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


