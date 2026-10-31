import pytest

from mjlab_microduck.tasks.quadruped_waist_recovery_env_cfg import (
    make_quadruped_waist_recovery_env_cfg,
)


def test_recovery_config_is_independent_and_registered():
    train = make_quadruped_waist_recovery_env_cfg()
    play = make_quadruped_waist_recovery_env_cfg(play=True)
    assert train is not play
    assert train.scene.entities["robot"] is not play.scene.entities["robot"]
    assert train.episode_length_s == pytest.approx(8.0)
    assert play.scene.num_envs == 1


def test_recovery_reset_and_rewards_allow_body_contact():
    cfg = make_quadruped_waist_recovery_env_cfg()
    assert cfg.events["reset_base"].func.__name__ == "set_random_ground_state"
    assert cfg.events["reset_base"].params["face_up_prob"] == pytest.approx(0.4)
    assert "fell_over" not in cfg.terminations
    assert "illegal_contact" not in cfg.terminations
    assert "recovery_success" in cfg.rewards
    assert cfg.rewards["nonfoot_contact"].weight == pytest.approx(-0.10)
    assert cfg.rewards["body_orientation_l2"].weight == pytest.approx(-0.1)


def test_recovery_has_no_velocity_or_gait_objective():
    cfg = make_quadruped_waist_recovery_env_cfg()
    assert cfg.commands["twist"].ranges.lin_vel_x == (0.0, 0.0)
    assert "track_linear_velocity" not in cfg.rewards
    assert "foot_gait" not in cfg.rewards
