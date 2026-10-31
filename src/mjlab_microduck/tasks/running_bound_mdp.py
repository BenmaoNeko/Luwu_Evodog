"""Pair coordination for commanded straight running on the waist quadruped."""

import torch
from mjlab.utils.lab_api.math import quat_apply_inverse

from mjlab_microduck.tasks.waist_gait_mdp import _gait_targets, _running_weight


def _running(env, command_name):
    command = env.command_manager.get_command(command_name)
    return (command[:, 0] > 0.4) & (command[:, 2].abs() < 1e-3)


def _pair_coordinates(env, asset_cfg, frame_cfg):
    data = env.scene[asset_cfg.name].data
    # Ordered sites: FR, FL, RR, RL. Each frame is at its hip-pair midpoint.
    feet = data.site_pos_w[:, asset_cfg.site_ids].reshape(env.num_envs, 2, 2, 3)
    origins = data.site_pos_w[:, frame_cfg.site_ids].unsqueeze(2)
    rotation = data.site_quat_w[:, frame_cfg.site_ids].unsqueeze(2).expand(-1, -1, 2, -1)
    offset = feet - origins
    return quat_apply_inverse(rotation, offset), rotation, offset


def running_foot_crossing(env, command_name, asset_cfg, frame_cfg, min_half_width):
    position, _, _ = _pair_coordinates(env, asset_cfg, frame_cfg)
    signed_y = position[..., 1] * position.new_tensor([-1.0, 1.0])
    cost = ((min_half_width - signed_y) / min_half_width).clamp(0.0, 1.0).mean(dim=(1, 2))
    running = _running(env, command_name)
    count = running.sum().clamp(min=1)
    crossed = (signed_y < 0).any(dim=2)
    env.extras["log"].update({
        "Metrics/running_front_cross_fraction": (crossed[:, 0] & running).sum() / count,
        "Metrics/running_rear_cross_fraction": (crossed[:, 1] & running).sum() / count,
    })
    return cost * running


def running_pair_contact_sync(env, command_name, sensor_name, tolerance_s):
    data = env.scene[sensor_name].data
    contact_age = data.current_contact_time.reshape(env.num_envs, 2, 2)
    air_age = data.current_air_time.reshape(env.num_envs, 2, 2)
    contact = contact_age > 0
    both_contact = contact.all(dim=2)
    both_air = (~contact).all(dim=2)
    split = contact[..., 0] != contact[..., 1]
    # Current stance/air ages encode touchdown/liftoff time differences.
    landing_gap = (contact_age[..., 0] - contact_age[..., 1]).abs() * both_contact
    liftoff_gap = (air_age[..., 0] - air_age[..., 1]).abs() * both_air
    split_age = torch.minimum(contact_age.max(dim=2).values, air_age.max(dim=2).values) * split
    gap = landing_gap + liftoff_gap + split_age
    cost = ((gap - tolerance_s - 1e-6) / tolerance_s).clamp(0.0, 1.0).mean(dim=1)
    running = _running(env, command_name)
    count = running.sum().clamp(min=1)
    env.extras["log"].update({
        "Metrics/running_front_contact_mismatch": (split[:, 0] & running).sum() / count,
        "Metrics/running_rear_contact_mismatch": (split[:, 1] & running).sum() / count,
        "Metrics/running_landing_gap_s": (landing_gap * running[:, None]).sum()
        / (both_contact & running[:, None]).sum().clamp(min=1),
        "Metrics/running_liftoff_gap_s": (liftoff_gap * running[:, None]).sum()
        / (both_air & running[:, None]).sum().clamp(min=1),
    })
    return cost * running


def running_pair_swing_sync(env, command_name, sensor_name, asset_cfg, frame_cfg,
                            position_tolerance, velocity_tolerance):
    position, rotation, offset = _pair_coordinates(env, asset_cfg, frame_cfg)
    data = env.scene[asset_cfg.name].data
    foot_velocity = data.site_lin_vel_w[:, asset_cfg.site_ids].reshape(env.num_envs, 2, 2, 3)
    torso_velocity = data.site_lin_vel_w[:, frame_cfg.site_ids].unsqueeze(2)
    torso_angular = data.site_ang_vel_w[:, frame_cfg.site_ids].unsqueeze(2).expand_as(offset)
    velocity = quat_apply_inverse(
        rotation, foot_velocity - torso_velocity - torch.cross(torso_angular, offset, dim=-1),
    )
    # Mirror lateral components; x/z must agree within each swinging pair.
    mirror = position.new_tensor([1.0, -1.0, 1.0])
    position_error = (position[:, :, 0] - position[:, :, 1] * mirror).abs()
    velocity_error = (velocity[:, :, 0] - velocity[:, :, 1] * mirror).abs()
    position_cost = ((position_error - position_tolerance).clamp(min=0) / position_tolerance).square()
    velocity_cost = ((velocity_error - velocity_tolerance).clamp(min=0) / velocity_tolerance).square()
    cost = 1.0 - torch.exp(-(position_cost + velocity_cost).mean(dim=2))
    contact = env.scene[sensor_name].data.current_contact_time.reshape(env.num_envs, 2, 2) > 0
    swinging = (~contact).all(dim=2)
    running = _running(env, command_name)
    count = (swinging & running[:, None]).sum().clamp(min=1)
    env.extras["log"]["Metrics/running_swing_position_error_m"] = (
        position_error.mean(dim=2) * swinging * running[:, None]
    ).sum() / count
    return (cost * swinging).mean(dim=1) * running


class RunningHipSplayPenalty:
    """Running hip deadband with formation-dependent, retained strength."""

    def __init__(self, cfg, env):
        self.formation = torch.zeros(env.num_envs, device=env.device)
        self.strength = torch.full_like(self.formation, 0.3)

    def reset(self, env_ids):
        # Curriculum strength and its filtered evidence survive episode resets.
        pass

    def __call__(self, env, command_name, asset_cfg, limit, excess_scale):
        running = _running(env, command_name)
        formation = env.reward_manager.get_term_cfg("foot_gait").func.formation
        alpha = 1.0 - torch.exp(self.formation.new_tensor(-env.step_dt / 5.0))
        self.formation += alpha * (formation - self.formation) * running
        target_strength = 0.3 + 1.2 * ((self.formation - 0.3) / 0.3).clamp(0.0, 1.0)
        self.strength = torch.maximum(self.strength, target_strength)
        data = env.scene[asset_cfg.name].data
        angle = (data.joint_pos[:, asset_cfg.joint_ids]
                 - data.default_joint_pos[:, asset_cfg.joint_ids]).abs()
        excess = (angle - limit).clamp(min=0)
        cost = (excess / excess_scale).square().mean(dim=1)
        count = running.sum().clamp(min=1)
        env.extras["log"].update({
            "Metrics/running_hip_angle_deg": (angle.mean(dim=1) * running).sum() / count * (180.0 / torch.pi),
            "Metrics/running_hip_excess_deg": (excess.mean(dim=1) * running).sum() / count * (180.0 / torch.pi),
            "Metrics/running_hip_outside_fraction": ((excess > 0).float().mean(dim=1) * running).sum() / count,
            "Curriculum/hip_penalty_strength": (self.strength * running).sum() / count,
            "Curriculum/hip_formation_smoothed": (self.formation * running).sum() / count,
        })
        return cost * self.strength * running


def running_waist_limits(env, command_name, asset_cfg, angle_limit, velocity_limit):
    data = env.scene[asset_cfg.name].data
    displacement = (data.joint_pos[:, asset_cfg.joint_ids]
                    - data.default_joint_pos[:, asset_cfg.joint_ids]).abs()
    angle_cost = ((displacement - angle_limit).clamp(min=0) / angle_limit).square()
    velocity_cost = ((data.joint_vel[:, asset_cfg.joint_ids].abs() - velocity_limit)
                     .clamp(min=0) / velocity_limit).square()
    return (angle_cost + velocity_cost).mean(dim=1) * _running(env, command_name)


def _running_knee_targets(env, asset_cfg, sensor_name, period, amplitude, swing_amplitude):
    data = env.scene[asset_cfg.name].data
    phase = (env.episode_length_buf * env.step_dt / period).unsqueeze(1)
    phase = (phase + phase.new_tensor([0.0, 0.0, 0.5, 0.5])) % 1.0
    # CAD forward kinematics: increasing each knee angle extends its leg.
    swing = phase >= 0.4
    contact = env.scene[sensor_name].data.current_contact_time > 0
    swing = swing & ~contact
    displacement = torch.where(swing,
                               -swing_amplitude * torch.sin(torch.pi * (phase - 0.4) / 0.6),
                               amplitude * torch.sin(torch.pi * phase / 0.4))
    # Actual support overrides the clock, including late liftoff and early touchdown.
    displacement = torch.where(contact, displacement.clamp(min=amplitude * 0.5), displacement)
    target = data.default_joint_pos[:, asset_cfg.joint_ids] + displacement
    return target, swing


def running_knee_phase(env, command_name, asset_cfg, sensor_name, period, amplitude, std,
                      swing_amplitude, swing_weight=1.0):
    data = env.scene[asset_cfg.name].data
    target, swing = _running_knee_targets(
        env, asset_cfg, sensor_name, period, amplitude, swing_amplitude)
    score = 1 / (1 + ((data.joint_pos[:, asset_cfg.joint_ids] - target) / std).square())
    running = _running(env, command_name)
    env.extras["log"]["Metrics/running_knee_phase_score"] = (
        score.mean(dim=1) * running).sum() / running.sum().clamp(min=1)
    active_swing = swing & running[:, None]
    count = active_swing.sum().clamp(min=1)
    env.extras["log"].update({
        "Metrics/running_knee_swing_score": (score * active_swing).sum() / count,
        "Metrics/running_knee_swing_error_deg": (
            (data.joint_pos[:, asset_cfg.joint_ids] - target).abs() * active_swing
        ).sum() / count * (180 / torch.pi),
    })
    return (score * torch.where(swing, swing_weight, 1.0)).mean(dim=1) * running


def running_knee_retraction(env, command_name, asset_cfg, sensor_name, period,
                            amplitude, swing_amplitude, error_scale):
    target, swing = _running_knee_targets(
        env, asset_cfg, sensor_name, period, amplitude, swing_amplitude)
    deficit = (env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids]
               - target).clamp(min=0)
    active = swing & _running(env, command_name)[:, None]
    env.extras["log"]["Metrics/running_knee_retraction_deficit_deg"] = (
        deficit * active).sum() / active.sum().clamp(min=1) * (180 / torch.pi)
    return ((deficit / error_scale).square() * active).mean(dim=1)


class RunningKneeInactivityPenalty:
    """Measure excursion over a full running cycle, including normal turnarounds."""

    def __init__(self, cfg, env):
        self.low = torch.full((env.num_envs, 4), torch.inf, device=env.device)
        self.high = torch.full_like(self.low, -torch.inf)
        self.elapsed = torch.zeros(env.num_envs, device=env.device)
        self.cost = torch.zeros_like(self.elapsed)
        self.excursion = torch.zeros_like(self.low)
        self.measured = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids):
        self.low[env_ids] = torch.inf
        self.high[env_ids] = -torch.inf
        self.elapsed[env_ids] = 0
        self.cost[env_ids] = 0
        self.excursion[env_ids] = 0
        self.measured[env_ids] = False

    def __call__(self, env, command_name, asset_cfg, period, min_excursion):
        running = _running(env, command_name)
        self.reset(~running)
        angle = env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids]
        self.low[running] = torch.minimum(self.low[running], angle[running])
        self.high[running] = torch.maximum(self.high[running], angle[running])
        self.elapsed += env.step_dt * running
        ready = running & (self.elapsed >= period - 1e-6)
        self.excursion[ready] = self.high[ready] - self.low[ready]
        self.cost[ready] = ((min_excursion - self.excursion[ready]).clamp(min=0)
                            / min_excursion).square().mean(dim=1)
        self.measured[ready] = True
        self.low[ready] = torch.inf
        self.high[ready] = -torch.inf
        self.elapsed[ready] = 0
        valid = running & self.measured
        count = valid.sum().clamp(min=1)
        env.extras["log"].update({
            "Metrics/running_knee_excursion_deg": (self.excursion.mean(dim=1) * valid).sum() / count * (180 / torch.pi),
            "Metrics/running_knee_inactive_fraction": ((self.excursion < min_excursion).float().mean(dim=1) * valid).sum() / count,
        })
        return self.cost * running


class RunningFlightReward:
    """Pay scheduled flight credit after a valid opposite-pair landing."""

    def __init__(self, cfg, env):
        self.last_support = torch.full((env.num_envs,), -1, device=env.device, dtype=torch.long)
        self.pending_landing = self.last_support.clone()
        self.last_landing = self.last_support.clone()
        self.support_age = torch.zeros(env.num_envs, device=env.device)
        self.flight_age = torch.zeros_like(self.support_age)
        self.landing_age = torch.zeros_like(self.support_age)
        self.flight_credit = torch.zeros_like(self.support_age)
        self.was_air = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

    def reset(self, env_ids):
        self.last_support[env_ids] = -1
        self.pending_landing[env_ids] = -1
        self.last_landing[env_ids] = -1
        self.support_age[env_ids] = 0
        self.flight_age[env_ids] = 0
        self.landing_age[env_ids] = 0
        self.flight_credit[env_ids] = 0
        self.was_air[env_ids] = False

    def __call__(self, env, command_name, sensor_name, period, tolerance_s=0.04):
        running = _running(env, command_name)
        self.reset(~running)
        contact = env.scene[sensor_name].data.current_contact_time > 0
        _, target, blend = _gait_targets(env, command_name, period, threshold=0.56)
        airborne = (~contact).all(dim=1)
        front = contact[:, 0] & contact[:, 1] & ~contact[:, 2] & ~contact[:, 3]
        rear = contact[:, 2] & contact[:, 3] & ~contact[:, 0] & ~contact[:, 1]
        group = torch.where(front, 0, torch.where(rear, 1, -1))
        support = running & (group >= 0) & (contact == target).all(dim=1)
        self.support_age += env.step_dt
        self.flight_age += env.step_dt
        self.landing_age += (~airborne) * env.step_dt

        start = running & airborne & ~self.was_air & (self.last_support >= 0)
        start &= self.support_age <= tolerance_s + env.step_dt + 1e-6
        self.pending_landing[start] = 1 - self.last_support[start]
        self.last_support[start] = -1
        self.flight_age[start] = 0
        self.landing_age[start] = 0
        self.flight_credit[start] = 0
        pending = self.pending_landing >= 0
        scheduled_air = (~target).all(dim=1)
        # Retain the existing tracking/attained-speed gates at the actual flight step.
        self.flight_credit += (
            pending * airborne * scheduled_air * running * _running_weight(env, command_name, blend)
        )
        landed = pending & support & (group == self.pending_landing)
        landed &= self.landing_age <= tolerance_s + env.step_dt + 1e-6
        landed &= self.flight_age <= period / 2 + 1e-6
        reward = torch.where(landed, self.flight_credit, 0.0)
        complete_cycle = landed & (reward > 0) & (group == 0) & (self.last_landing == 1)
        self.last_landing[landed & (reward > 0)] = group[landed & (reward > 0)]
        expired = pending & ((self.flight_age > period / 2 + 1e-6)
                             | (self.landing_age > tolerance_s + env.step_dt + 1e-6)
                             | ((group >= 0) & (group != self.pending_landing))
                             | contact.all(dim=1))
        self.last_landing[expired & ~landed] = -1
        clear = landed | expired
        self.pending_landing[clear] = -1
        self.flight_credit[clear] = 0
        self.last_support[support] = group[support]
        self.support_age[support] = 0
        self.was_air = airborne & running
        count = running.sum().clamp(min=1)
        env.extras["log"].update({
            "Metrics/running_airborne_fraction": (airborne & running).sum() / count,
            "Metrics/running_valid_flight_landing": ((reward > 0) & running).sum() / count,
            "Metrics/running_complete_cycle": complete_cycle.sum() / count,
        })
        return reward * running


def running_invalid_reward(env, command_name, asset_cfg, frame_cfg, sensor_name,
                           positive_terms, period, tolerance_s, min_half_width,
                           residual_fraction):
    """Remove positive credit after all other terms, preserving penalty magnitudes."""
    position, _, _ = _pair_coordinates(env, asset_cfg, frame_cfg)
    signed_y = position[..., 1] * position.new_tensor([-1.0, 1.0])
    separated = (signed_y >= min_half_width).all(dim=(1, 2))
    data = env.scene[sensor_name].data
    contact_age = data.current_contact_time.reshape(env.num_envs, 2, 2)
    air_age = data.current_air_time.reshape(env.num_envs, 2, 2)
    paired_contact = contact_age > 0
    same_contact = paired_contact.all(dim=2)
    same_air = (~paired_contact).all(dim=2)
    split = paired_contact[..., 0] != paired_contact[..., 1]
    gap = ((contact_age[..., 0] - contact_age[..., 1]).abs() * same_contact
           + (air_age[..., 0] - air_age[..., 1]).abs() * same_air
           + torch.minimum(contact_age.max(dim=2).values, air_age.max(dim=2).values) * split)
    synchronized = (gap <= tolerance_s + 1e-6).all(dim=1)
    contact = data.current_contact_time > 0
    phase = (env.episode_length_buf * env.step_dt / period).unsqueeze(1)
    phase = (phase + phase.new_tensor([0.0, 0.0, 0.5, 0.5])) % 1.0
    target = phase < 0.4
    # Only mismatched feet near a scheduled transition receive timing tolerance.
    boundary_distance = torch.minimum(torch.minimum(phase, 1.0 - phase), (phase - 0.4).abs())
    phase_ok = ((contact == target) | (boundary_distance * period <= tolerance_s + 1e-6)).all(dim=1)
    phase_ok &= ~contact.all(dim=1)
    valid = separated & synchronized & phase_ok
    running = _running(env, command_name)
    manager = env.reward_manager
    # RewardManager fills these unscaled rates in order; this term must be last.
    indices = [manager.active_terms.index(name) for name in positive_terms]
    positive_rate = manager._step_reward[:, indices].clamp(min=0).sum(dim=1)
    invalid = running & ~valid
    count = running.sum().clamp(min=1)
    env.extras["log"].update({
        "Metrics/running_hard_valid_fraction": (valid & running).sum() / count,
        "Metrics/running_phase_valid_fraction": (phase_ok & running).sum() / count,
        "Metrics/running_separation_valid_fraction": (separated & running).sum() / count,
        "Metrics/running_sync_valid_fraction": (synchronized & running).sum() / count,
    })
    return invalid * ((1.0 - residual_fraction) * positive_rate + 1.0)
