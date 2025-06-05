from __future__ import annotations

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import CommandManager, CurriculumManager, TerminationManager

from .reward_manager import RewardManager


class CustomManagerBasedRLEnv(ManagerBasedRLEnv):
    """Manager-based RL environment using the custom reward manager."""

    def load_managers(self):
        """Load and initialize environment managers."""
        # Command manager first so observation manager can access commands
        self.command_manager: CommandManager = CommandManager(self.cfg.commands, self)

        # Load default managers (observation, action, event, recorder, viewer)
        super().load_managers()

        # Termination, reward and curriculum managers
        self.termination_manager = TerminationManager(self.cfg.terminations, self)
        self.reward_manager = RewardManager(self.cfg.rewards, self)
        self.curriculum_manager = CurriculumManager(self.cfg.curriculum, self)

        # Setup Gym spaces
        self._configure_gym_env_spaces()

        # Trigger startup events if available
        if "startup" in self.event_manager.available_modes:
            self.event_manager.apply(mode="startup")


