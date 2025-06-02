from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG, CUBOID_MARKER_CFG, RED_ARROW_X_MARKER_CFG
from isaaclab.utils.math import compute_pose_error, quat_from_euler_xyz, combine_frame_transforms
import math

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .commands_cfg import *


class EePoseCommand(CommandTerm):
    cfg: EePoseCommandCfg

    def __init__(self, cfg: EePoseCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.env = env

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.body_idx = self.robot.find_bodies(cfg.ee_name)[0][0]

        self.pose_command_b = torch.zeros(self.num_envs, 7, device=self.device)
        self.pose_command_b[:, 3] = 1.0
        self.pose_command_w = torch.zeros_like(self.pose_command_b)
        
        self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["orientation_error"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        msg = "UniformPoseCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        return msg
    
    @property
    def command(self) -> torch.Tensor:
        return self.pose_command_b

    def _update_metrics(self):
        self.pose_command_w[:, :3] = self.env.scene.env_origins + self.pose_command_b[:,:3]
        self.pose_command_w[:, 3:] = self.pose_command_b[:, 3:7]

        pos_error, rot_error = compute_pose_error(
            self.pose_command_w[:, :3],
            self.pose_command_w[:, 3:],
            self.robot.data.body_state_w[:, self.body_idx, :3],
            self.robot.data.body_state_w[:, self.body_idx, 3:7],
        )
        self.metrics["position_error"] = torch.norm(pos_error, dim=-1)
        self.metrics["orientation_error"] = torch.norm(rot_error, dim=-1)

    def _resample_command(self, env_ids: Sequence[int]):
        dis_xy = torch.empty(len(env_ids), device=self.device).uniform_(*self.cfg.ranges.dis_xy)
        theta = torch.empty(len(env_ids), device=self.device).uniform_(0, 2 * math.pi)
        pos_z = torch.empty(len(env_ids), device=self.device).uniform_(*self.cfg.ranges.pos_z)

        self.pose_command_b[env_ids, 0] = dis_xy * torch.cos(theta)
        self.pose_command_b[env_ids, 1] = dis_xy * torch.sin(theta)
        self.pose_command_b[env_ids, 2] = pos_z
        
        euler_angles = torch.zeros_like(self.pose_command_b[env_ids, :3])
        euler_angles[:, 0].uniform_(*self.cfg.ranges.roll)
        euler_angles[:, 1].uniform_(*self.cfg.ranges.pitch)
        euler_angles[:, 2].uniform_(*self.cfg.ranges.yaw)
        quat = quat_from_euler_xyz(euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2])
        self.pose_command_b[env_ids, 3:] = quat

    def _update_command(self):
        pass

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_pose_visualizer"):
                marker_cfg = FRAME_MARKER_CFG.copy()
                marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
                
                marker_cfg.prim_path = "/Visuals/Command/goal_pose"
                self.goal_pose_visualizer = VisualizationMarkers(marker_cfg)
                
                marker_cfg.prim_path = "/Visuals/Command/body_pose"
                self.body_pose_visualizer = VisualizationMarkers(marker_cfg)
            
            self.goal_pose_visualizer.set_visibility(True)
            self.body_pose_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer.set_visibility(False)
                self.body_pose_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return
        self.goal_pose_visualizer.visualize(self.pose_command_w[:, :3], self.pose_command_w[:, 3:])
        
        body_pose_w = self.robot.data.body_state_w[:, self.body_idx]
        self.body_pose_visualizer.visualize(body_pose_w[:, :3], body_pose_w[:, 3:7])


class BaseHeightCommand(CommandTerm):
    cfg: BaseHeightCommandCfg

    def __init__(self, cfg: BaseHeightCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.body_idx = self.robot.find_bodies("base_link")[0][0]

        self.height_command = torch.zeros(self.num_envs, 1, device=self.device)
        
        self.metrics["height_error"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        msg = "UniformPoseCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        return msg

    @property
    def command(self) -> torch.Tensor:
        return self.height_command

    def _update_metrics(self):
        height_error = self.robot.data.body_state_w[:, self.body_idx, 2] - self.height_command
        self.metrics["height_error"] = torch.norm(height_error, dim=-1)

    def _resample_command(self, env_ids: Sequence[int]):
        r = torch.empty(len(env_ids), device=self.device)
        self.height_command[env_ids, 0] = r.uniform_(*self.cfg.ranges.base_height)

    def _update_command(self):
        pass

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_pose_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.markers["cuboid"].size = (0.5, 0.5, 0.001)
                marker_cfg.markers["cuboid"].visual_material=sim_utils.GlassMdlCfg(mdl_path="OmniGlass_Opacity.mdl", glass_color=(1.0, 0.0, 0.0), frosting_roughness=0.5)
                marker_cfg.prim_path = "/Visuals/Command/goal_height"
                self.goal_height_visualizer = VisualizationMarkers(marker_cfg)
            self.goal_height_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_height_visualizer"):
                self.goal_height_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return
        base_posi_w = self.robot.data.body_state_w[:, self.body_idx][:, :3]
        base_posi_w[:, 2] = self.height_command[:, 0]
        self.goal_height_visualizer.visualize(base_posi_w)


class EePoseMovingCommand(CommandTerm):
    cfg: EePoseMovingCommandCfg

    def __init__(self, cfg: EePoseCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.env = env

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.body_idx = self.robot.find_bodies(cfg.ee_name)[0][0]

        self.start_point_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.end_point_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.pose_command_b = torch.zeros(self.num_envs, 7, device=self.device)
        self.pose_command_b[:, 3] = 1.0
        self.pose_command_w = torch.zeros_like(self.pose_command_b)
        self.max_episode_len = self.env.max_episode_length
        
        self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["orientation_error"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        msg = "UniformPoseCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        return msg
    
    @property
    def command(self) -> torch.Tensor:
        return self.pose_command_b

    def _update_metrics(self):
        b = self.env.episode_length_buf / self.max_episode_len
        self.pose_command_b[:, :3] = self.start_point_b + ((self.end_point_b - self.start_point_b) * (self.env.episode_length_buf / self.max_episode_len).unsqueeze(1))
        self.pose_command_w[:, :3] = self.pose_command_b[:, :3] + self.env.scene.env_origins

        pos_error, rot_error = compute_pose_error(
            self.pose_command_w[:, :3],
            self.pose_command_w[:, 3:],
            self.robot.data.body_state_w[:, self.body_idx, :3],
            self.robot.data.body_state_w[:, self.body_idx, 3:7],
        )
        self.metrics["position_error"] = torch.norm(pos_error, dim=-1)
        self.metrics["orientation_error"] = torch.norm(rot_error, dim=-1)


    def _resample_command(self, env_ids: Sequence[int]):
        self.start_point_b = torch.tensor(self.cfg.ranges.start_point, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        self.end_point_b = torch.tensor(self.cfg.ranges.end_point, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        self.pose_command_b[env_ids, :3] = self.start_point_b[env_ids]
        self.pose_command_w[env_ids, :3] = self.pose_command_b[env_ids, :3] + self.env.scene.env_origins[env_ids]
        

        euler_angles = torch.zeros_like(self.pose_command_w[env_ids, :3])
        euler_angles[:, 0].uniform_(*self.cfg.ranges.roll)
        euler_angles[:, 1].uniform_(*self.cfg.ranges.pitch)
        euler_angles[:, 2].uniform_(*self.cfg.ranges.yaw)
        self.pose_command_b[env_ids, 3:] = quat_from_euler_xyz(euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2])
        self.pose_command_w[env_ids, 3:] = self.pose_command_b[env_ids, 3:]
        
    def _update_command(self):
        pass

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_pose_visualizer"):
                marker_cfg = FRAME_MARKER_CFG.copy()
                goal_marker_cfg = RED_ARROW_X_MARKER_CFG.copy()
                marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
                goal_marker_cfg.markers["arrow"].scale = (0.1, 0.1, 0.5)
                
                goal_marker_cfg.prim_path = "/Visuals/Command/goal_pose"
                self.goal_pose_visualizer = VisualizationMarkers(goal_marker_cfg)
                # marker_cfg.prim_path = "/Visuals/Command/goal_pose"
                # self.goal_pose_visualizer = VisualizationMarkers(marker_cfg)
                
                marker_cfg.prim_path = "/Visuals/Command/body_pose"
                self.body_pose_visualizer = VisualizationMarkers(marker_cfg)
            
            self.goal_pose_visualizer.set_visibility(True)
            self.body_pose_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer.set_visibility(False)
                self.body_pose_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return
        self.goal_pose_visualizer.visualize(self.pose_command_w[:, :3])
        
        body_pose_w = self.robot.data.body_state_w[:, self.body_idx]
        self.body_pose_visualizer.visualize(body_pose_w[:, :3], body_pose_w[:, 3:7])