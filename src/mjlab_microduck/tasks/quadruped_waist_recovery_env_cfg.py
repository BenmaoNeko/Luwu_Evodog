"""Independent fall-recovery task for the 13-joint waist quadruped."""

from copy import deepcopy

from mjlab.envs import mdp as env_mdp
from mjlab.managers import EventTermCfg, RewardTermCfg, TerminationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from mjlab_microduck.robot.quadruped_waist_constants import (
    STAND_HEIGHT, WAIST_QUADRUPED_ROBOT_CFG,
)
from mjlab_microduck.tasks import mdp as recovery_mdp
from mjlab_microduck.tasks.quadruped_velocity_env_cfg import (
    QuadrupedRlCfg, make_quadruped_velocity_env_cfg,
)


_BODY = SceneEntityCfg("robot", body_names=("base_link",))
_WAIST = SceneEntityCfg("robot", body_names=("w2",))
_LEGS = SceneEntityCfg("robot", joint_names=(r"^joint[0-9]+$",))


def make_quadruped_waist_recovery_env_cfg(play: bool = False):
    """Recover from prone/supine/sitting starts, then hold a true stand."""
    cfg = make_quadruped_velocity_env_cfg(play=play)
    cfg.scene.entities["robot"] = deepcopy(WAIST_QUADRUPED_ROBOT_CFG)
    cfg.episode_length_s = 8.0
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None

    command = cfg.commands["twist"]
    command.heading_command = False
    command.ranges.lin_vel_x = (0.0, 0.0)
    command.ranges.lin_vel_y = (0.0, 0.0)
    command.ranges.ang_vel_z = (0.0, 0.0)
    command.ranges.heading = None
    command.resampling_time_range = (10.0, 10.0)
    command.rel_standing_envs = 1.0

    cfg.events["reset_base"] = EventTermCfg(
        func=recovery_mdp.set_random_ground_state,
        mode="reset",
        params={
            "face_down_prob": 0.4,
            "face_up_prob": 0.4,
            "sitting_prob": 0.2,
            "standing_prob": 0.0,
            "prone_z_min": 0.08,
            "prone_z_max": 0.11,
            "sitting_z_min": 0.08,
            "sitting_z_max": 0.10,
            "standing_z_min": STAND_HEIGHT - 0.004,
            "standing_z_max": STAND_HEIGHT + 0.004,
            "face_up_roll_max": 0.35,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    cfg.events["reset_robot_joints"].params["position_range"] = (0.0, 0.0)
    cfg.events["reset_robot_joints"].params["velocity_range"] = (0.0, 0.0)
    cfg.events.pop("push_robot", None)
    cfg.actions["joint_pos"] = recovery_mdp.FreezeThenJointPositionActionCfg(
        entity_name="robot", actuator_names=(r"joint[0-9]+",),
        scale=0.25, use_default_offset=True, freeze_s=1.0,
    )

    for name in (
        "track_linear_velocity", "track_angular_velocity", "foot_gait",
        "foot_clearance", "foot_slip", "soft_landing", "stand_still", "pose",
    ):
        cfg.rewards.pop(name, None)
    cfg.rewards["upright_linear"] = RewardTermCfg(
        func=recovery_mdp.body_upright_linear, weight=2.0, params={"asset_cfg": _BODY},
    )
    cfg.rewards["upright_progress"] = RewardTermCfg(
        func=recovery_mdp.upright_progress, weight=5.0, params={"asset_cfg": _BODY},
    )
    cfg.rewards["side_roll_progress"] = RewardTermCfg(
        func=recovery_mdp.side_roll_progress, weight=10.0, params={"asset_cfg": _BODY},
    )
    cfg.rewards["side_to_upright_progress"] = RewardTermCfg(
        func=recovery_mdp.side_to_upright_progress, weight=8.0, params={"asset_cfg": _BODY},
    )
    cfg.rewards["upright_sharp"] = RewardTermCfg(
        func=recovery_mdp.body_upright_gaussian, weight=1.0,
        params={"asset_cfg": _BODY, "std": 0.30},
    )
    cfg.rewards["waist_upright"] = RewardTermCfg(
        func=recovery_mdp.body_upright_linear, weight=1.0, params={"asset_cfg": _WAIST},
    )
    cfg.rewards["height_stand"] = RewardTermCfg(
        func=recovery_mdp.height_target_gaussian, weight=2.0,
        params={"target_height": STAND_HEIGHT, "std": 0.025, "asset_cfg": _BODY},
    )
    cfg.rewards["height_stand_l1"] = RewardTermCfg(
        func=recovery_mdp.height_l1_penalty, weight=3.0,
        params={"target_height": STAND_HEIGHT, "asset_cfg": _BODY},
    )
    cfg.rewards["pose_stand_legs"] = RewardTermCfg(
        func=recovery_mdp.pose_target_match, weight=2.0,
        params={"std": 0.35, "joint_indices": list(range(13)), "target_overrides": None},
    )
    cfg.rewards["standing_composite"] = RewardTermCfg(
        func=recovery_mdp.standing_composite_score, weight=5.0,
        params={"target_height": STAND_HEIGHT, "height_std": 0.03,
                "upright_std": 0.35, "pose_std": 0.40,
                "joint_indices": list(range(13)), "target_overrides": None,
                "asset_cfg": _BODY},
    )
    cfg.rewards["com_upward_velocity"] = RewardTermCfg(
        func=recovery_mdp.com_upward_velocity, weight=0.5,
        params={"max_height": STAND_HEIGHT + 0.01, "max_vz": 0.35, "asset_cfg": _BODY},
    )
    cfg.rewards["recovery_success"] = RewardTermCfg(
        func=recovery_mdp.recovery_success, weight=20.0,
        params={"fallen_tilt_deg": 40.0, "min_fallen_s": 0.4,
                "up_tilt_deg": None, "up_z": STAND_HEIGHT - 0.01,
                "foot_sensor_name": "feet_ground_contact",
                "joint_indices": list(range(13)), "pose_tol": 0.60,
                "asset_cfg": _BODY},
    )
    cfg.rewards["fallen_state_penalty"] = RewardTermCfg(
        func=recovery_mdp.fallen_state_penalty, weight=-0.5,
        params={"asset_cfg": _BODY, "gate_tilt_above_deg": 40.0,
                "release_tilt_below_deg": 25.0, "release_z_above": STAND_HEIGHT - 0.01},
    )
    cfg.rewards["body_ang_vel"] = RewardTermCfg(
        func=recovery_mdp.body_ang_vel_at_height, weight=-0.03,
        params={"height_low": STAND_HEIGHT - 0.02, "height_high": STAND_HEIGHT,
                "tilt_full_deg": 20.0, "tilt_zero_deg": 45.0, "asset_cfg": _BODY},
    )
    # Keep the luwu recovery recipe's light orientation shaping; the robot is
    # intentionally allowed to lie on its body while discovering a recovery.
    cfg.rewards["body_orientation_l2"].weight = -0.1
    cfg.rewards["action_rate_l2"].weight = -0.05
    cfg.rewards["nonfoot_contact"].weight = -0.10

    cfg.terminations.pop("fell_over", None)
    cfg.terminations.pop("illegal_contact", None)
    cfg.terminations.pop("base_too_low", None)
    cfg.terminations["nan_state"] = TerminationTermCfg(
        func=recovery_mdp.robot_state_is_nan,
        params={"sensor_names": ("feet_ground_contact",)},
    )
    cfg.curriculum.clear()

    if play:
        cfg.scene.num_envs = 1
        cfg.observations["actor"].enable_corruption = False
        for name in ("base_com", "encoder_bias", "foot_friction"):
            cfg.events.pop(name, None)
    return cfg


QuadrupedWaistRecoveryRlCfg = deepcopy(QuadrupedRlCfg)
QuadrupedWaistRecoveryRlCfg.experiment_name = "quadruped_waist_recovery"
QuadrupedWaistRecoveryRlCfg.run_name = "quadruped_waist_recovery"
QuadrupedWaistRecoveryRlCfg.max_iterations = 10000
