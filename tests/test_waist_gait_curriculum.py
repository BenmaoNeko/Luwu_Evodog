"""Tensor-level behavioral checks; run on the remote training server."""

from types import SimpleNamespace
import unittest

import torch

from mjlab_microduck.tasks.waist_gait_mdp import (
    ProgressiveVelocityCommandCfg, GaitTransitionReward, track_linear_velocity,
)


def make_env(n=1):
    data = SimpleNamespace(
        root_link_lin_vel_b=torch.zeros(n, 3),
        root_link_ang_vel_b=torch.zeros(n, 3),
        projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0]]).repeat(n, 1),
    )
    command = torch.zeros(n, 3)
    env = SimpleNamespace(
        num_envs=n, device="cpu", step_dt=0.02,
        episode_length_buf=torch.full((n,), 5),
        scene={
            "robot": SimpleNamespace(data=data),
            "feet": SimpleNamespace(data=SimpleNamespace(current_contact_time=torch.zeros(n, 4))),
        },
        command_manager=SimpleNamespace(get_command=lambda name: command),
        extras={"log": {}},
    )
    return env, command


class WaistGaitTests(unittest.TestCase):
    def test_running_diagonal_support_loses_tracking_and_has_small_cost(self):
        from mjlab_microduck.tasks.waist_gait_mdp import running_diagonal_support
        env, command = make_env(4)
        command[:, 0] = torch.tensor([0.6, 0.6, 0.6, 0.3])
        env.scene["robot"].data.root_link_lin_vel_b.copy_(command)
        env.scene["feet"].data.current_contact_time[:] = torch.tensor([
            [0, 1, 1, 0], [1, 0, 0, 1], [1, 0, 1, 0], [0, 1, 1, 0],
        ])
        tracking = track_linear_velocity(env, "twist", 0.3, diagonal_tracking_scale=0.025, sensor_name="feet")
        torch.testing.assert_close(tracking, torch.tensor([0.025, 0.025, 1.0, 1.0]))
        torch.testing.assert_close(running_diagonal_support(env, "twist", "feet"), torch.tensor([1., 1., 0., 0.]))
        command[:, 2] = 0.2
        torch.testing.assert_close(running_diagonal_support(env, "twist", "feet"), torch.zeros(4))

    def test_running_gait_gives_diagonal_no_credit_even_below_actual_speed(self):
        env, command = make_env(2)
        command[:, 0] = 0.6
        env.scene["robot"].data.root_link_lin_vel_b[:, 0] = torch.tensor([0.6, 0.3])
        env.scene["feet"].data.current_contact_time[:] = torch.tensor([0, 1, 1, 0])
        torch.testing.assert_close(GaitTransitionReward(None, env)(env, "twist", "feet", 0.4, 0.56), torch.zeros(2))

    def test_speed_waits_at_06_until_bound_is_formed(self):
        from mjlab.managers.reward_manager import RewardManager
        from mjlab_microduck.tasks.quadruped_waist_velocity_env_cfg import make_quadruped_waist_velocity_env_cfg
        env, _ = make_env(2)
        cfg = make_quadruped_waist_velocity_env_cfg()
        cfg.commands["twist"].rel_forward_envs = 1.0
        cfg.commands["twist"].require_bound_for_speed = True
        cfg.commands["twist"].speed_increment = 0.1
        cfg.commands["twist"].tracking_hold_s = 0.3
        cfg.commands["twist"].min_level_time_s = 0.0
        env.reward_manager = RewardManager({"foot_gait": cfg.rewards["foot_gait"]}, env)
        env.reward_manager.get_term_cfg("foot_gait").func.stage[1] = 2
        term = cfg.commands["twist"].build(env)
        term.time_left[:] = 10.0
        term.is_forward_env[:] = True
        term.speed_cap[:] = 0.6
        term.vel_command_b[:, 0] = 0.6
        term.target_command[:, 0] = 0.6
        env.scene["robot"].data.root_link_lin_vel_b[:, 0] = 0.6
        for _ in range(16):
            term.compute(0.02)
        torch.testing.assert_close(term.speed_cap, torch.tensor([0.6, 0.7]))

    def test_gait_term_works_with_reward_manager_and_partial_reset(self):
        from mjlab.managers.reward_manager import RewardManager
        from mjlab_microduck.tasks.quadruped_waist_velocity_env_cfg import make_quadruped_waist_velocity_env_cfg
        env, command = make_env(2)
        env.max_episode_length_s = 20.0
        env.scene["feet_ground_contact"] = env.scene["feet"]
        command[:, 0] = 0.6
        env.scene["robot"].data.root_link_lin_vel_b[:, 0] = 0.6
        env.scene["feet"].data.current_contact_time[:] = torch.tensor([1, 1, 0, 0])
        cfg = make_quadruped_waist_velocity_env_cfg().rewards["foot_gait"]
        manager = RewardManager({"foot_gait": cfg}, env)
        gait = manager.get_term_cfg("foot_gait").func
        self.assertIsInstance(gait, GaitTransitionReward)
        gait.stage[0] = 2
        torch.testing.assert_close(manager.compute(env.step_dt), torch.tensor([0.24, 0.15]))
        manager.reset(torch.tensor([0]))
        torch.testing.assert_close(gait.stage, torch.tensor([2, 0]))
        torch.testing.assert_close(gait.running_steps, torch.tensor([0, 1]))

    def test_gait_stages_require_full_window_and_use_30_and_60_percent(self):
        env, command = make_env(4)
        command[:, 0] = 0.6
        env.scene["robot"].data.root_link_lin_vel_b[:, 0] = 0.6
        env.scene["feet"].data.current_contact_time[:] = torch.tensor([0, 1, 1, 0])
        gait = GaitTransitionReward(None, env)
        gait.running_steps[:] = 98
        gait.support_steps[:] = 98
        gait.matched_steps[:] = torch.tensor([29, 30, 59, 60])
        gait(env, "twist", "feet", 0.4, 0.56)
        torch.testing.assert_close(gait.stage, torch.zeros(4, dtype=torch.long))
        gait(env, "twist", "feet", 0.4, 0.56)
        torch.testing.assert_close(gait.stage, torch.tensor([0, 1, 1, 2]))
        torch.testing.assert_close(gait.formation, torch.tensor([0.29, 0.30, 0.59, 0.60]))
        # The remaining bound partial-match reward is separate from trot credit.
        env.scene["feet"].data.current_contact_time[:] = torch.tensor([1, 1, 0, 0])
        score = gait(env, "twist", "feet", 0.4, 0.56)
        torch.testing.assert_close(score, torch.tensor([2.5, 3.0, 3.0, 4.0]))
        gait.reset(torch.arange(4))
        torch.testing.assert_close(gait.stage, torch.tensor([0, 1, 1, 2]))
        torch.testing.assert_close(gait.running_steps, torch.zeros(4, dtype=torch.long))
        # Low-speed walking keeps its full trot reward, even after running mastery.
        command[:, 0] = 0.3
        env.scene["robot"].data.root_link_lin_vel_b[:, 0] = 0.3
        env.scene["feet"].data.current_contact_time[:] = torch.tensor([0, 1, 1, 0])
        torch.testing.assert_close(gait(env, "twist", "feet", 0.4, 0.56), torch.ones(4))

    def test_formation_counts_support_not_flight_or_turning(self):
        env, command = make_env(3)
        command[:, 0] = 0.6
        env.scene["robot"].data.root_link_lin_vel_b[:, 0] = 0.6
        gait = GaitTransitionReward(None, env)
        for step in range(100):
            # Sample between contact boundaries so both support groups are observed.
            env.episode_length_buf.fill_(2 * step + 1)
            phase = ((2 * step + 1) * 0.02 / 0.8) % 1.0
            front = phase < 0.4
            rear = (phase + 0.5) % 1.0 < 0.4
            contacts = env.scene["feet"].data.current_contact_time
            contacts[0] = torch.tensor([front, rear, front, rear])
            contacts[1] = torch.tensor([0, 1, 1, 0])
            contacts[2] = 0
            gait(env, "twist", "feet", 0.8, 0.56)
        torch.testing.assert_close(gait.stage, torch.tensor([2, 0, 0]))
        torch.testing.assert_close(gait.formation, torch.tensor([1.0, 0.0, 0.0]))
        command[:, 2] = 0.2
        for _ in range(110):
            gait(env, "twist", "feet", 0.8, 0.56)
        torch.testing.assert_close(gait.running_steps, torch.zeros(3, dtype=torch.long))

    def test_relaxed_promotion_still_rejects_standing(self):
        env, _ = make_env(2)
        cfg = ProgressiveVelocityCommandCfg(
            entity_name="robot", resampling_time_range=(8.0, 12.0), rel_forward_envs=1.0,
            ranges=ProgressiveVelocityCommandCfg.Ranges(
                lin_vel_x=(-0.2, 2.0), lin_vel_y=(-0.15, 0.15), ang_vel_z=(-0.8, 0.8),
            ),
        )
        term = cfg.build(env)
        term.reset(torch.arange(2))
        term.vel_command_b[:, 0] = 0.2
        term.target_command[:, 0] = 0.2
        env.scene["robot"].data.root_link_lin_vel_b[0, 0] = 0.1
        env.scene["robot"].data.root_link_ang_vel_b[0, 2] = 0.2
        for _ in range(16):
            term.compute(0.02)
        torch.testing.assert_close(term.speed_cap, torch.tensor([0.3, 0.2]))

    def test_low_high_transition_and_attained_speed(self):
        env, command = make_env()
        robot = env.scene["robot"].data
        contact = env.scene["feet"].data.current_contact_time

        def score(speed, attained, feet):
            command[0] = torch.tensor([speed, 0.0, 0.0])
            robot.root_link_lin_vel_b[0] = torch.tensor([attained, 0.0, 0.0])
            contact[0] = torch.tensor(feet)
            return GaitTransitionReward(None, env)(env, "twist", "feet", 0.4, 0.56).item()

        trot = [0, 1, 1, 0]
        bound = [1, 1, 0, 0]
        self.assertAlmostEqual(score(0.3, 0.3, trot), 1.0)
        self.assertAlmostEqual(score(0.3, 0.3, bound), 0.5)
        self.assertAlmostEqual(score(0.6, 0.6, bound), 2.5)
        self.assertAlmostEqual(score(0.6, 0.6, trot), 0.0)
        self.assertEqual(score(0.45, 0.45, trot), 0.0)
        self.assertGreater(score(0.45, 0.45, bound), score(0.45, 0.45, trot))
        self.assertEqual(score(0.6, 0.3, trot), 0.0)

    def test_translation_only_and_no_standing_bonus(self):
        env, command = make_env()
        command[0, 1] = 0.2
        env.scene["robot"].data.root_link_lin_vel_b.copy_(command)
        contact = env.scene["feet"].data.current_contact_time
        contact[0] = torch.tensor([0, 1, 1, 0])
        self.assertAlmostEqual(GaitTransitionReward(None, env)(env, "twist", "feet", 0.4, 0.56).item(), 1.0)
        command[0, 2] = 0.1
        self.assertEqual(GaitTransitionReward(None, env)(env, "twist", "feet", 0.4, 0.56).item(), 0.0)
        command[0, 2] = 0.0
        contact.fill_(1)
        self.assertEqual(GaitTransitionReward(None, env)(env, "twist", "feet", 0.4, 0.56).item(), 0.0)
        env.episode_length_buf.zero_()
        self.assertEqual(GaitTransitionReward(None, env)(env, "twist", "feet", 0.4, 0.56).item(), 0.0)

    def test_tracking_remains_dense_and_rewards_matching(self):
        env, command = make_env(4)
        command[:, 0] = 2.0
        env.scene["robot"].data.root_link_lin_vel_b[:, 0] = torch.tensor([0.0, 1.0, 1.8, 2.0])
        reward = track_linear_velocity(env, "twist", 0.3)
        self.assertGreater(reward[0].item(), 0.01)
        self.assertTrue((reward[1:] > reward[:-1]).all().item())
        self.assertEqual(reward[-1].item(), 1.0)

    def test_tracking_does_not_penalize_vertical_velocity(self):
        env, command = make_env(2)
        command[:, 0] = 0.6
        env.scene["robot"].data.root_link_lin_vel_b[:, 0] = 0.6
        env.scene["robot"].data.root_link_lin_vel_b[1, 2] = 1.0
        torch.testing.assert_close(track_linear_velocity(env, "twist", 0.3), torch.ones(2))

    def test_running_flight_matches_gait_only_in_flight_windows(self):
        from mjlab_microduck.tasks.waist_gait_mdp import running_flight
        env, command = make_env()
        command[0, 0] = 0.6
        env.scene["robot"].data.root_link_lin_vel_b[0, 0] = 0.6
        for step in (9, 19):
            env.episode_length_buf.fill_(step)
            self.assertAlmostEqual(GaitTransitionReward(None, env)(env, "twist", "feet", 0.4, 0.56).item(), 2.5)
            self.assertAlmostEqual(running_flight(env, "twist", "feet", 0.4).item(), 1.0)
        env.episode_length_buf.fill_(5)
        self.assertEqual(running_flight(env, "twist", "feet", 0.4).item(), 0.0)
        env.episode_length_buf.fill_(9)
        env.scene["feet"].data.current_contact_time[0, 0] = 1.0
        self.assertEqual(running_flight(env, "twist", "feet", 0.4).item(), 0.0)

    def test_running_guides_require_running_straight_and_upright(self):
        from mjlab_microduck.tasks.waist_gait_mdp import running_flight, running_waist_motion
        from mjlab.managers.scene_entity_config import SceneEntityCfg
        env, command = make_env(5)
        command[:, 0] = torch.tensor([0.6, 0.3, 0.6, 0.6, 0.6])
        command[3, 2] = 0.2
        data = env.scene["robot"].data
        data.root_link_lin_vel_b[:, 0] = torch.tensor([0.6, 0.3, 0.2, 0.6, 0.6])
        data.projected_gravity_b[4, 2] = 1.0
        data.joint_pos = torch.zeros(5, 1)
        data.default_joint_pos = torch.zeros(5, 1)
        env.episode_length_buf.fill_(9)
        asset = SceneEntityCfg("robot", joint_names=("joint13",))
        asset.joint_ids = [0]
        for reward in (
            running_flight(env, "twist", "feet", 0.4),
            running_waist_motion(env, "twist", asset, 0.4, 0.25, 0.15),
        ):
            self.assertGreater(reward[0].item(), 0.0)
            torch.testing.assert_close(reward[1:], torch.zeros(4))

    def test_waist_guide_prefers_home_relative_alternating_motion(self):
        from mjlab_microduck.tasks.waist_gait_mdp import running_waist_motion
        from mjlab.managers.scene_entity_config import SceneEntityCfg
        env, command = make_env(3)
        command[:, 0] = 0.6
        data = env.scene["robot"].data
        data.root_link_lin_vel_b[:, 0] = 0.6
        data.default_joint_pos = torch.full((3, 1), 0.1)
        asset = SceneEntityCfg("robot", joint_names=("joint13",))
        asset.joint_ids = [0]
        for step, target in ((5, 0.25), (15, -0.25)):
            env.episode_length_buf.fill_(step)
            data.joint_pos = data.default_joint_pos + torch.tensor([[target], [0.0], [-target]])
            scores = running_waist_motion(env, "twist", asset, 0.4, 0.25, 0.15)
            self.assertAlmostEqual(scores[0].item(), 1.0)
            self.assertGreater(scores[0].item(), scores[1].item())
            self.assertGreater(scores[1].item(), scores[2].item())

    def test_waist_task_reward_configuration(self):
        import math
        from mjlab_microduck.tasks.quadruped_waist_velocity_env_cfg import make_quadruped_waist_velocity_env_cfg
        from mjlab_microduck.tasks.quadruped_velocity_env_cfg import make_quadruped_velocity_env_cfg
        cfg = make_quadruped_waist_velocity_env_cfg()
        self.assertEqual(cfg.commands["twist"].ranges.lin_vel_x, (-0.2, 1.2))
        self.assertFalse(cfg.commands["twist"].require_bound_for_speed)
        self.assertEqual(cfg.commands["twist"].bound_learning_speed, 0.6)
        self.assertEqual(cfg.rewards["running_diagonal_support"].weight, -0.3)
        self.assertEqual(cfg.rewards["track_linear_velocity"].params["diagonal_tracking_scale"], 0.025)
        self.assertEqual(make_quadruped_waist_velocity_env_cfg(play=True).commands["twist"].ranges.lin_vel_x, (-0.2, 1.2))
        self.assertNotIn("body_orientation_l2", cfg.rewards)
        self.assertNotIn("body_ang_vel", cfg.rewards)
        self.assertNotIn("body_ang_vel", make_quadruped_waist_velocity_env_cfg(play=True).rewards)
        self.assertNotIn("body_orientation_l2", make_quadruped_waist_velocity_env_cfg(play=True).rewards)
        self.assertAlmostEqual(cfg.rewards["pose"].params["std_walking"][r"^joint13$"], math.radians(12))
        self.assertAlmostEqual(cfg.rewards["pose"].params["std_running"][r"^joint13$"], math.radians(20))
        self.assertEqual(cfg.rewards["running_flight"].weight, 2.25)
        self.assertEqual(cfg.rewards["running_waist_motion"].weight, 0.6)
        base = make_quadruped_velocity_env_cfg()
        self.assertEqual(base.rewards["body_orientation_l2"].weight, -3.0)
        self.assertEqual(base.rewards["body_ang_vel"].weight, -0.08)
        self.assertNotIn("running_flight", base.rewards)
        self.assertNotIn("running_waist_motion", base.rewards)
        self.assertNotIn("running_diagonal_support", base.rewards)

    def test_progress_requires_tracking_and_reset_never_jumps(self):
        env, _ = make_env(2)
        cfg = ProgressiveVelocityCommandCfg(
            entity_name="robot", resampling_time_range=(8.0, 12.0), rel_forward_envs=1.0,
            ranges=ProgressiveVelocityCommandCfg.Ranges(
                lin_vel_x=(-0.2, 2.0), lin_vel_y=(-0.15, 0.15), ang_vel_z=(-0.8, 0.8),
            ),
        )
        term = cfg.build(env)
        ids = torch.arange(2)
        term.reset(ids)
        term.compute(0.0)
        self.assertTrue((term.command == 0).all().item())
        for _ in range(200):
            term.compute(0.02)
        torch.testing.assert_close(term.speed_cap, torch.tensor([0.2, 0.2]))
        for _ in range(200):
            env.scene["robot"].data.root_link_lin_vel_b[0] = term.command[0]
            previous = term.command.clone()
            term.compute(0.02)
            self.assertLessEqual((term.command - previous).abs().max().item(), 0.005001)
        self.assertGreater(term.speed_cap[0].item(), 0.4)
        self.assertAlmostEqual(term.speed_cap[1].item(), 0.2)
        previous = term.command.clone()
        term._resample(ids)
        torch.testing.assert_close(term.command, previous)
        earned = term.speed_cap.clone()
        term.reset(ids)
        term.compute(0.0)
        self.assertTrue((term.command == 0).all().item())
        torch.testing.assert_close(term.speed_cap, earned)


if __name__ == "__main__":
    unittest.main()
