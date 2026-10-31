"""Tracking-gated speed progression and speed-dependent waist quadruped gaits."""

from dataclasses import dataclass
import math

import torch
from mjlab.tasks.velocity.mdp.velocity_command import (
    UniformVelocityCommand, UniformVelocityCommandCfg,
)


class ProgressiveVelocityCommand(UniformVelocityCommand):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.speed_cap = torch.full((self.num_envs,), cfg.initial_speed, device=self.device)
        self.tracking_time = torch.zeros_like(self.speed_cap)
        self.level_time = torch.zeros_like(self.speed_cap)
        self.target_command = torch.zeros_like(self.vel_command_b)
        self.command_dt = 0.0
        self.window_steps = max(1, math.ceil(cfg.tracking_window_s / env.step_dt))
        self.window_index = 0
        self.tracked_window = torch.zeros((self.num_envs, self.window_steps), device=self.device)
        self.eligible_window = torch.zeros_like(self.tracked_window)
        self.window_age = torch.zeros_like(self.speed_cap)

    def reset(self, env_ids):
        extras = super().reset(env_ids)
        # Retain earned speed levels, but restart every episode from rest.
        self.vel_command_b[env_ids] = 0.0
        self.vel_command_w[env_ids] = 0.0
        self.tracking_time[env_ids] = 0.0
        self.tracked_window[env_ids] = 0.0
        self.eligible_window[env_ids] = 0.0
        self.window_age[env_ids] = 0.0
        return extras

    def _resample_command(self, env_ids):
        forward = torch.rand(len(env_ids), device=self.device) < self.cfg.rel_forward_envs
        self.is_forward_env[env_ids] = forward
        target = torch.empty((len(env_ids), 3), device=self.device)
        low = self.cfg.ranges.lin_vel_x[0]
        target[:, 0] = low + torch.rand(len(env_ids), device=self.device) * (self.speed_cap[env_ids] - low)
        target[:, 1].uniform_(*self.cfg.ranges.lin_vel_y)
        target[:, 2].uniform_(*self.cfg.ranges.ang_vel_z)
        target[forward, 0] = self.speed_cap[env_ids][forward] - torch.rand(
            len(env_ids), device=self.device,
        )[forward] * self.cfg.forward_speed_window
        target[forward, 1:] = 0.0
        self.target_command[env_ids] = target
        self.tracking_time[env_ids] = 0.0

    def compute(self, dt):
        self.command_dt = dt
        super().compute(dt)

    def _update_command(self):
        dt = self.command_dt
        actual = self.robot.data.root_link_lin_vel_b
        error = torch.linalg.vector_norm(actual[:, :2] - self.vel_command_b[:, :2], dim=1)
        at_target = (self.vel_command_b[:, 0] - self.target_command[:, 0]).abs() < 0.01
        straight = self.is_forward_env & (self.vel_command_b[:, 2].abs() < 1e-3)
        tracked = (
            straight & at_target
            & (error < 0.10 + 0.1 * self.speed_cap)
        )
        self.tracking_time = torch.where(tracked, self.tracking_time + dt, 0.0)
        self.level_time += dt
        ceiling = torch.full_like(self.speed_cap, self.cfg.ranges.lin_vel_x[1])
        if self.cfg.require_bound_for_speed:
            stage = self._env.reward_manager.get_term_cfg("foot_gait").func.stage
            ceiling[stage < 2] = self.cfg.bound_learning_speed
        demote = torch.zeros_like(tracked)
        success_rate = tracked.float()
        if self.cfg.tracking_window_s > 0:
            eligible = straight & at_target
            self.tracked_window[:, self.window_index] = tracked.float()
            self.eligible_window[:, self.window_index] = eligible.float()
            self.window_index = (self.window_index + 1) % self.window_steps
            self.window_age += dt
            eligible_count = self.eligible_window.sum(dim=1)
            success_rate = self.tracked_window.sum(dim=1) / eligible_count.clamp(min=1)
            ready = ((self.window_age >= self.cfg.tracking_window_s)
                     & (eligible_count >= 0.6 * self.window_steps) & eligible)
            tracking_ready = ready & (success_rate >= self.cfg.promotion_success_rate)
            demote = (ready & (success_rate < self.cfg.demotion_success_rate)
                      & (self.level_time >= self.cfg.min_level_time_s)
                      & (self.speed_cap > self.cfg.initial_speed))
        else:
            tracking_ready = self.tracking_time >= self.cfg.tracking_hold_s
        promote = (
            tracking_ready
            & (self.level_time >= self.cfg.min_level_time_s)
            & (self.speed_cap < ceiling)
        )
        self.speed_cap[promote] = torch.minimum(self.speed_cap[promote] + self.cfg.speed_increment, ceiling[promote])
        self.speed_cap[demote] = (self.speed_cap[demote] - self.cfg.speed_increment).clamp(min=self.cfg.initial_speed)
        changed = promote | demote
        self.target_command[changed, 0] = self.speed_cap[changed] - torch.rand(
            self.num_envs, device=self.device,
        )[changed] * self.cfg.forward_speed_window
        self.tracking_time[changed] = 0.0
        self.level_time[changed] = 0.0
        self.tracked_window[changed] = 0.0
        self.eligible_window[changed] = 0.0
        self.window_age[changed] = 0.0

        # Resampling changes targets, never jumps the command seen by the policy.
        self.vel_command_b[:, :2] += (self.target_command[:, :2] - self.vel_command_b[:, :2]).clamp(
            min=-self.cfg.acceleration * dt, max=self.cfg.acceleration * dt,
        )
        self.vel_command_b[:, 2] += (self.target_command[:, 2] - self.vel_command_b[:, 2]).clamp(
            min=-0.5 * dt, max=0.5 * dt,
        )
        self._env.extras["log"].update({
            "Curriculum/speed_cap_mean": self.speed_cap.mean(),
            "Curriculum/speed_cap_max": self.speed_cap.max(),
            "Curriculum/level_time_mean": self.level_time.mean(),
            "Curriculum/tracking_time_mean": self.tracking_time.mean(),
            "Curriculum/tracking_success_rate_mean": success_rate.mean(),
            "Curriculum/promote_fraction": promote.float().mean(),
            "Curriculum/demote_fraction": demote.float().mean(),
            "Metrics/command_vx_mean": self.vel_command_b[:, 0].mean(),
            "Metrics/actual_vx_mean": actual[:, 0].mean(),
            "Metrics/straight_fraction": straight.float().mean(),
        })


@dataclass(kw_only=True)
class ProgressiveVelocityCommandCfg(UniformVelocityCommandCfg):
    initial_speed: float = 0.2
    forward_speed_window: float = 0.0
    speed_increment: float = 0.1
    acceleration: float = 0.25
    tracking_hold_s: float = 0.3
    min_level_time_s: float = 0.0
    require_bound_for_speed: bool = False
    bound_learning_speed: float = 0.6
    tracking_window_s: float = 0.0
    promotion_success_rate: float = 0.8
    demotion_success_rate: float = 0.35

    def build(self, env):
        return ProgressiveVelocityCommand(self, env)


def track_linear_velocity(env, command_name: str, std: float,
                          diagonal_tracking_scale: float = 1.0, sensor_name: str = "feet_ground_contact",
                          running_reward_scale: float = 1.0):
    """Dense tracking reward, including useful feedback outside a narrow kernel."""
    command = env.command_manager.get_command(command_name)
    actual = env.scene["robot"].data.root_link_lin_vel_b
    error = (command[:, :2] - actual[:, :2]).square().sum(dim=1)
    reward = 1.0 / (1.0 + error / std**2)
    running = (command[:, 0] > 0.4) & (command[:, 2].abs() < 1e-3)
    reward *= torch.where(running, running_reward_scale, 1.0)
    if diagonal_tracking_scale != 1.0:
        contact = env.scene[sensor_name].data.current_contact_time > 0
        running = (command[:, 0] > 0.4) & (command[:, 2].abs() < 1e-3)
        reward *= torch.where(running & _diagonal_support(contact), diagonal_tracking_scale, 1.0)
    return reward


def running_diagonal_support(env, command_name: str, sensor_name: str):
    command = env.command_manager.get_command(command_name)
    contact = env.scene[sensor_name].data.current_contact_time > 0
    running = (command[:, 0] > 0.4) & (command[:, 2].abs() < 1e-3)
    return (running & _diagonal_support(contact)).float()


def _diagonal_support(contact):
    # Sensor model order FR, FL, RR, RL: either diagonal, regardless of phase.
    return ((contact[:, 0] & contact[:, 3] & ~contact[:, 1] & ~contact[:, 2])
            | (contact[:, 1] & contact[:, 2] & ~contact[:, 0] & ~contact[:, 3]))


class GaitTransitionReward:
    """Reward walking trot and running bound, with support-based bound stages."""

    def __init__(self, cfg, env):
        self.stage = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.running_steps = torch.zeros_like(self.stage)
        self.support_steps = torch.zeros_like(self.stage)
        self.matched_steps = torch.zeros_like(self.stage)
        self.formation = torch.zeros(env.num_envs, device=env.device)
        self.bound_factors = torch.tensor([2.5, 3.0, 4.0], device=env.device)

    def reset(self, env_ids):
        # Learned stages survive episode resets; incomplete measurements do not.
        self.running_steps[env_ids] = 0
        self.support_steps[env_ids] = 0
        self.matched_steps[env_ids] = 0

    def __call__(self, env, command_name: str, sensor_name: str, period: float,
                 threshold: float, formation_window_s: float = 2.0,
                 forming_threshold: float = 0.3, formed_threshold: float = 0.6):
        trot, bound, blend = _gait_targets(env, command_name, period, threshold)
        contact = env.scene[sensor_name].data.current_contact_time > 0
        command = env.command_manager.get_command(command_name)
        actual = env.scene["robot"].data.root_link_lin_vel_b
        straight = command[:, 2].abs() < 1e-3
        moving = torch.linalg.vector_norm(command[:, :2], dim=1) > 0.1
        running = (command[:, 0] > 0.4) & straight
        error = torch.linalg.vector_norm(command[:, :2] - actual[:, :2], dim=1)
        valid = running & (blend > 0) & (error < 0.10 + 0.1 * command[:, 0])
        support = valid & bound.any(dim=1)
        matched = (contact == bound).all(dim=1)
        self.running_steps += valid.long()
        self.support_steps += support.long()
        self.matched_steps += (support & matched).long()
        evaluate = self.running_steps >= math.ceil(formation_window_s / env.step_dt)
        rate = self.matched_steps / self.support_steps.clamp(min=1)
        self.formation[evaluate] = rate[evaluate]
        next_stage = (rate >= forming_threshold).long() + (rate >= formed_threshold).long()
        self.stage[evaluate] = torch.maximum(self.stage[evaluate], next_stage[evaluate])
        self.reset(evaluate)

        # Partial foot matching gives feedback before the complete bound is learned.
        mixed_contact = contact.any(dim=1) & (~contact).any(dim=1)
        trot_score = (contact == trot).float().mean(dim=1) * mixed_contact
        # Main credit requires both complete pairs; one wrong foot gets only guidance.
        matches = (contact == bound).sum(dim=1)
        bound_score = torch.where(matches == 4, 1.0, torch.where(matches == 3, 0.1, 0.0))
        bound_score *= mixed_contact | ((~contact).all(dim=1) & (~bound).all(dim=1))
        trot_scale = (~running).float()
        bound_scale = self.bound_factors[self.stage] * running * _running_progress(env, command_name)
        score = trot_scale * trot_score + bound_scale * bound_score
        env.extras["log"].update({
            "Curriculum/bound_formation_mean": self.formation.mean(),
            "Curriculum/gait_stage_mean": self.stage.float().mean(),
            "Curriculum/trot_factor_mean": trot_scale.mean(),
            "Curriculum/bound_factor_mean": bound_scale.mean(),
        })
        return score * straight * moving * track_linear_velocity(env, command_name, std=0.3)


def running_flight(env, command_name: str, sensor_name: str, period: float):
    """Reward all-feet flight only during the bound's scheduled flight windows."""
    _, stance, blend = _gait_targets(env, command_name, period, threshold=0.56)
    airborne = (env.scene[sensor_name].data.current_contact_time <= 0).all(dim=1)
    return airborne * (~stance).all(dim=1) * _running_weight(env, command_name, blend)


def running_waist_motion(env, command_name: str, asset_cfg, period: float, amplitude: float, std: float):
    """Weak home-relative phase guidance; amplitude is a target, not a limit."""
    _, _, blend = _gait_targets(env, command_name, period, threshold=0.56)
    robot = env.scene[asset_cfg.name]
    phase = env.episode_length_buf * env.step_dt / period
    target = amplitude * blend * torch.sin(2 * torch.pi * phase)
    displacement = robot.data.joint_pos[:, asset_cfg.joint_ids] - robot.data.default_joint_pos[:, asset_cfg.joint_ids]
    score = torch.exp(-((displacement - target.unsqueeze(1)) / std).square().mean(dim=1))
    return score * _running_weight(env, command_name, blend)


def _running_weight(env, command_name, blend):
    command = env.command_manager.get_command(command_name)
    straight = command[:, 2].abs() < 1e-3
    upright = env.scene["robot"].data.projected_gravity_b[:, 2] < -0.9
    return (blend * straight * upright * track_linear_velocity(env, command_name, std=0.3)
            * _running_progress(env, command_name))


def _running_progress(env, command_name):
    command = env.command_manager.get_command(command_name)
    actual = env.scene["robot"].data.root_link_lin_vel_b[:, 0].clamp(min=0)
    tracking_ratio = (actual / command[:, 0].clamp(min=0.4)).clamp(max=1.0)
    progress = (actual / 0.8).clamp(max=1.0) * tracking_ratio.square()
    running = (command[:, 0] > 0.4) & (command[:, 2].abs() < 1e-3)
    count = running.sum().clamp(min=1)
    env.extras["log"].update({
        "Metrics/running_forward_progress": (progress * running).sum() / count,
        "Metrics/running_actual_vx_mean": (actual * running).sum() / count,
        "Metrics/running_command_vx_mean": (command[:, 0] * running).sum() / count,
    })
    return progress


def _gait_targets(env, command_name, period, threshold):
    command = env.command_manager.get_command(command_name)
    actual = env.scene["robot"].data.root_link_lin_vel_b
    phase = (env.episode_length_buf * env.step_dt / period).unsqueeze(1)
    speed = torch.minimum(command[:, 0], actual[:, 0])
    blend = ((speed - 0.4) / 0.1).clamp(0.0, 1.0)
    # Contact sensors follow model traversal, FR, FL, RR, RL (not FOOT_BODIES).
    trot_offsets = torch.tensor([0.5, 0.0, 0.0, 0.5], device=env.device)
    bound_offsets = torch.tensor([0.0, 0.0, 0.5, 0.5], device=env.device)
    trot = (phase + trot_offsets) % 1.0 < threshold
    bound = (phase + bound_offsets) % 1.0 < 0.4
    env.extras["log"]["Metrics/bound_blend_mean"] = blend.mean()
    return trot, bound, blend
