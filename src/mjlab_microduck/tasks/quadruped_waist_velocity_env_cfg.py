"""Isolated waist variant: same hardware, flat scene and reset events."""
from copy import deepcopy
import math

from mjlab.managers import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from mjlab_microduck.robot.quadruped_waist_constants import WAIST_QUADRUPED_ROBOT_CFG
from mjlab_microduck.tasks.quadruped_velocity_env_cfg import (
    QuadrupedRlCfg, make_quadruped_velocity_env_cfg,
)
from mjlab_microduck.tasks import running_bound_mdp, waist_gait_mdp


def make_quadruped_waist_velocity_env_cfg(play: bool = False):
    cfg = make_quadruped_velocity_env_cfg(play=play)
    cfg.scene.entities["robot"] = deepcopy(WAIST_QUADRUPED_ROBOT_CFG)
    cfg.rewards.pop("body_orientation_l2")
    cfg.rewards.pop("body_ang_vel")
    cfg.commands["twist"].ranges.lin_vel_x = (-0.2, 1.4)
    cfg.commands["twist"] = waist_gait_mdp.ProgressiveVelocityCommandCfg(
        **vars(cfg.commands["twist"]),
        initial_speed=0.6,
        forward_speed_window=0.1,
        speed_increment=0.05,
        tracking_hold_s=5.0,
        min_level_time_s=60.0,
        require_bound_for_speed=False,
        tracking_window_s=5.0,
        promotion_success_rate=0.8,
        demotion_success_rate=0.35,
    )
    cfg.commands["twist"].rel_forward_envs = 0.8
    cfg.commands["twist"].resampling_time_range = (8.0, 12.0)
    cfg.rewards["track_linear_velocity"].func = waist_gait_mdp.track_linear_velocity
    cfg.rewards["track_linear_velocity"].weight = 10.0
    cfg.rewards["track_linear_velocity"].params["std"] = 0.3
    cfg.rewards["track_linear_velocity"].params["diagonal_tracking_scale"] = 0.025
    cfg.rewards["track_linear_velocity"].params["running_reward_scale"] = 2.0
    cfg.rewards["action_rate_l2"].weight = -0.60
    cfg.rewards["running_diagonal_support"] = RewardTermCfg(
        func=waist_gait_mdp.running_diagonal_support, weight=-0.3,
        params={"command_name": "twist", "sensor_name": "feet_ground_contact"},
    )
    cfg.rewards["foot_gait"].func = waist_gait_mdp.GaitTransitionReward
    cfg.rewards["foot_gait"].weight = 3.0
    cfg.rewards["foot_gait"].params = {
        "command_name": "twist", "sensor_name": "feet_ground_contact",
        "period": 0.4, "threshold": 0.56,
        "formation_window_s": 2.0, "forming_threshold": 0.3, "formed_threshold": 0.6,
    }
    for mode, std in (("standing", 0.05), ("walking", math.radians(6)), ("running", math.radians(10))):
        cfg.rewards["pose"].params[f"std_{mode}"][r"^joint13$"] = std
    cfg.rewards["running_waist_motion"] = RewardTermCfg(
        func=waist_gait_mdp.running_waist_motion, weight=0.6,
        params={
            "command_name": "twist", "period": 0.8,
            "asset_cfg": SceneEntityCfg("robot", joint_names=("^joint13$",)),
            "amplitude": math.radians(7.5), "std": math.radians(5),
        },
    )
    cfg.rewards["running_flight"] = RewardTermCfg(
        func=running_bound_mdp.RunningFlightReward, weight=2.25,
        params={"command_name": "twist", "sensor_name": "feet_ground_contact",
                "period": 0.4, "tolerance_s": 0.04},
    )
    # Pair geometry and actual contact sensor model order are FR, FL, RR, RL.
    feet = SceneEntityCfg("robot", site_names=(
        "r1_3_foot", "L1_3_foot", "r2_3_foot", "L2_3_foot",
    ), preserve_order=True)
    frames = SceneEntityCfg("robot", site_names=("front_gait_frame", "rear_gait_frame"),
                            preserve_order=True)
    cfg.rewards["running_foot_crossing"] = RewardTermCfg(
        func=running_bound_mdp.running_foot_crossing, weight=-3.0,
        params={"command_name": "twist", "asset_cfg": feet, "frame_cfg": frames,
                "min_half_width": 0.019},
    )
    cfg.rewards["running_pair_contact_sync"] = RewardTermCfg(
        func=running_bound_mdp.running_pair_contact_sync, weight=-3.0,
        params={"command_name": "twist", "sensor_name": "feet_ground_contact",
                "tolerance_s": 0.04},
    )
    cfg.rewards["running_pair_swing_sync"] = RewardTermCfg(
        func=running_bound_mdp.running_pair_swing_sync, weight=-0.5,
        params={"command_name": "twist", "sensor_name": "feet_ground_contact",
                "asset_cfg": feet, "frame_cfg": frames,
                "position_tolerance": 0.01, "velocity_tolerance": 0.1},
    )
    cfg.rewards["running_hip_splay"] = RewardTermCfg(
        func=running_bound_mdp.RunningHipSplayPenalty, weight=-2.0,
        params={"command_name": "twist",
                "asset_cfg": SceneEntityCfg("robot", joint_names=("^joint(1|4|7|10)$",)),
                "limit": math.radians(15), "excess_scale": math.radians(10)},
    )
    cfg.rewards["running_waist_limits"] = RewardTermCfg(
        func=running_bound_mdp.running_waist_limits, weight=-0.5,
        params={"command_name": "twist",
                "asset_cfg": SceneEntityCfg("robot", joint_names=("^joint13$",)),
                "angle_limit": math.radians(10),
                "velocity_limit": math.radians(7.5) * 2 * math.pi / 0.8},
    )
    knees = SceneEntityCfg("robot", joint_names=("^joint3$", "^joint9$", "^joint6$", "^joint12$"),
                           preserve_order=True)
    cfg.rewards["running_knee_phase"] = RewardTermCfg(
        func=running_bound_mdp.running_knee_phase, weight=3.0,
        params={"command_name": "twist", "asset_cfg": knees,
                "sensor_name": "feet_ground_contact", "period": 0.4,
                "amplitude": math.radians(20), "std": math.radians(10),
                "swing_amplitude": math.radians(45), "swing_weight": 2.0},
    )
    cfg.rewards["running_knee_retraction"] = RewardTermCfg(
        func=running_bound_mdp.running_knee_retraction, weight=-1.0,
        params={"command_name": "twist", "asset_cfg": knees,
                "sensor_name": "feet_ground_contact", "period": 0.4,
                "amplitude": math.radians(20), "swing_amplitude": math.radians(45),
                "error_scale": math.radians(30)},
    )
    cfg.rewards["running_knee_inactivity"] = RewardTermCfg(
        func=running_bound_mdp.RunningKneeInactivityPenalty, weight=-0.5,
        params={"command_name": "twist", "asset_cfg": knees, "period": 0.4,
                "min_excursion": math.radians(15)},
    )
    # Append after every positive term so the gate cancels this step's actual credit.
    cfg.rewards["running_invalid_reward"] = RewardTermCfg(
        func=running_bound_mdp.running_invalid_reward, weight=-1.0,
        params={"command_name": "twist", "asset_cfg": feet, "frame_cfg": frames,
                "sensor_name": "feet_ground_contact",
                "positive_terms": tuple(name for name, term in cfg.rewards.items() if term.weight > 0),
                "period": 0.4, "tolerance_s": 0.04, "min_half_width": 0.019,
                "residual_fraction": 0.02},
    )
    return cfg


QuadrupedWaistRlCfg = deepcopy(QuadrupedRlCfg)
QuadrupedWaistRlCfg.experiment_name = "quadruped_waist_velocity"
QuadrupedWaistRlCfg.run_name = "quadruped_waist"
