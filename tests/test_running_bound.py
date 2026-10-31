"""Run-only coordination checks using actual reward functions and contact traces."""

from types import SimpleNamespace
import unittest

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply

from mjlab_microduck.tasks import running_bound_mdp as mdp
from mjlab_microduck.tasks.waist_gait_mdp import GaitTransitionReward


def make_env(n=1):
    command = torch.zeros(n, 3)
    command[:, 0] = 0.6
    position = torch.tensor([
        [0.02, -0.038, -0.12], [0.02, 0.038, -0.12],
        [-0.02, -0.038, -0.12], [-0.02, 0.038, -0.12],
        [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
    ]).repeat(n, 1, 1)
    quaternion = torch.zeros(n, 6, 4)
    quaternion[:, :, 0] = 1
    data = SimpleNamespace(
        site_pos_w=position, site_quat_w=quaternion,
        site_lin_vel_w=torch.zeros(n, 6, 3), site_ang_vel_w=torch.zeros(n, 6, 3),
        root_link_lin_vel_b=command.clone(), projected_gravity_b=torch.tensor([0., 0., -1.]).repeat(n, 1),
    )
    contact = SimpleNamespace(current_contact_time=torch.zeros(n, 4), current_air_time=torch.zeros(n, 4))
    env = SimpleNamespace(
        num_envs=n, device="cpu", step_dt=0.02, episode_length_buf=torch.full((n,), 5),
        command_manager=SimpleNamespace(get_command=lambda name: command),
        scene={"robot": SimpleNamespace(data=data), "feet": SimpleNamespace(data=contact)},
        extras={"log": {}},
    )
    feet = SceneEntityCfg("robot", site_ids=[0, 1, 2, 3])
    frames = SceneEntityCfg("robot", site_ids=[4, 5])
    return env, command, feet, frames


class RunningBoundTests(unittest.TestCase):
    def test_crossing_and_swing_penalties_are_run_only(self):
        env, command, feet, frames = make_env(6)
        command[:, 0] = torch.tensor([0.6, 0.4, 0.3, -0.6, 0.6, 0.6])
        command[4, 2] = 0.1
        env.scene["robot"].data.site_pos_w[:, 0, 1] = 0.04
        crossing = mdp.running_foot_crossing(env, "twist", feet, frames, 0.019)
        swing = mdp.running_pair_swing_sync(env, "twist", "feet", feet, frames, 0.01, 0.1)
        for cost in (crossing, swing):
            self.assertGreater(cost[0].item(), 0)
            torch.testing.assert_close(cost[1:5], torch.zeros(4))
            self.assertGreater(cost[5].item(), 0)  # Even when achieved speed is zero.
        env.scene["robot"].data.root_link_lin_vel_b[5] = 0
        self.assertGreater(mdp.running_foot_crossing(env, "twist", feet, frames, 0.019)[5].item(), 0)

    def test_rear_frame_rotation_does_not_penalize_waist_motion(self):
        env, _, feet, frames = make_env()
        data = env.scene["robot"].data
        rear_ids = [2, 3, 5]
        q = torch.tensor([0.9800666, 0.0, 0.1986693, 0.0]).repeat(1, 3, 1)
        data.site_pos_w[:, rear_ids] = quat_apply(q, data.site_pos_w[:, rear_ids])
        data.site_quat_w[:, rear_ids] = q
        torch.testing.assert_close(mdp.running_foot_crossing(env, "twist", feet, frames, 0.019), torch.zeros(1))
        torch.testing.assert_close(mdp.running_pair_swing_sync(env, "twist", "feet", feet, frames, 0.01, 0.1), torch.zeros(1))

    def test_swing_sync_uses_local_frames_and_rotating_frame_velocity(self):
        env, _, feet, frames = make_env(2)
        data = env.scene["robot"].data
        q = torch.tensor([0.9238795, 0.0, 0.3826834, 0.0])
        rotation = q.repeat(2, 6, 1)
        translation = torch.tensor([1.0, 2.0, 3.0])
        offset = quat_apply(rotation, data.site_pos_w)
        data.site_pos_w = offset + translation
        data.site_quat_w = rotation
        angular = torch.tensor([0.1, 0.2, 0.3]).expand_as(offset)
        linear = torch.tensor([0.4, 0.5, 0.6])
        data.site_ang_vel_w[:] = angular
        data.site_lin_vel_w = linear + torch.cross(angular, offset, dim=-1)
        torch.testing.assert_close(mdp.running_foot_crossing(env, "twist", feet, frames, 0.019), torch.zeros(2))
        torch.testing.assert_close(mdp.running_pair_swing_sync(env, "twist", "feet", feet, frames, 0.01, 0.1), torch.zeros(2))
        data.site_lin_vel_w[1, 0, 0] += 1.0
        cost = mdp.running_pair_swing_sync(env, "twist", "feet", feet, frames, 0.01, 0.1)
        self.assertEqual(cost[0].item(), 0)
        self.assertGreater(cost[1].item(), 0)

    def test_contact_timing_tolerance_and_run_only_gate(self):
        env, command, _, _ = make_env(4)
        contact = env.scene["feet"].data
        # Front pair has touchdown gaps 40ms / 80ms; rear pair is synchronous.
        contact.current_contact_time[:] = torch.tensor([
            [0.10, 0.06, 0.0, 0.0], [0.14, 0.06, 0.0, 0.0],
            [0.14, 0.06, 0.0, 0.0], [0.14, 0.06, 0.0, 0.0],
        ])
        contact.current_air_time[:, [2, 3]] = 0.1
        command[2, 0] = 0.3
        command[3, 2] = 0.2
        cost = mdp.running_pair_contact_sync(env, "twist", "feet", 0.04)
        self.assertEqual(cost[0].item(), 0)
        self.assertGreater(cost[1].item(), 0)
        torch.testing.assert_close(cost[2:], torch.zeros(2))
        contact.current_contact_time.zero_()
        contact.current_air_time[:] = torch.tensor([0.14, 0.06, 0.10, 0.10])
        self.assertGreater(mdp.running_pair_contact_sync(env, "twist", "feet", 0.04)[0].item(), 0)
        # One foot down / one up: grace period applies to the latest transition.
        contact.current_contact_time[:] = torch.tensor([0.02, 0.0, 0.0, 0.0])
        contact.current_air_time[:] = torch.tensor([0.0, 0.1, 0.1, 0.1])
        self.assertEqual(mdp.running_pair_contact_sync(env, "twist", "feet", 0.04)[0].item(), 0)
        contact.current_contact_time[:, 0] = 0.10
        self.assertGreater(mdp.running_pair_contact_sync(env, "twist", "feet", 0.04)[0].item(), 0)

    def test_paired_gait_keeps_walking_and_reduces_single_foot_credit(self):
        env, command, _, _ = make_env(4)
        contact = env.scene["feet"].data.current_contact_time
        contact[:] = torch.tensor([[1, 1, 0, 0], [1, 0, 0, 0], [0, 1, 1, 0], [1, 0, 0, 0]])
        command[3, 0] = 0.3
        env.scene["robot"].data.root_link_lin_vel_b.copy_(command)
        score = GaitTransitionReward(None, env)(env, "twist", "feet", 0.4, 0.56)
        torch.testing.assert_close(score, torch.tensor([2.5, 0.25, 0.0, 0.25]))

    def test_flight_requires_support_and_correct_opposite_landing(self):
        env, command, _, _ = make_env(6)
        term = mdp.RunningFlightReward(None, env)
        contact = env.scene["feet"].data.current_contact_time
        command[4, 0] = 0.3
        command[5, 2] = 0.2
        # Front support at phase .35, then air across the boundary, rear at .50.
        env.episode_length_buf.fill_(7)
        contact[:] = torch.tensor([1, 1, 0, 0])
        contact[1] = 0  # No preceding support.
        term(env, "twist", "feet", 0.4)
        contact.zero_()
        for step in (8, 9):
            env.episode_length_buf.fill_(step)
            torch.testing.assert_close(term(env, "twist", "feet", 0.4), torch.zeros(6))
        term.reset(torch.tensor([3]))
        env.episode_length_buf.fill_(10)
        contact[:] = torch.tensor([0, 0, 1, 1])
        contact[2] = torch.tensor([1, 1, 0, 0])  # Wrong landing group.
        reward = term(env, "twist", "feet", 0.4)
        self.assertGreater(reward[0].item(), 0)
        torch.testing.assert_close(reward[1:], torch.zeros(5))
        torch.testing.assert_close(term(env, "twist", "feet", 0.4), torch.zeros(6))

    def test_flight_expiry_and_walking_clear_history(self):
        env, command, _, _ = make_env(2)
        term = mdp.RunningFlightReward(None, env)
        contact = env.scene["feet"].data.current_contact_time
        contact[:] = torch.tensor([1, 1, 0, 0])
        env.episode_length_buf.fill_(7)
        term(env, "twist", "feet", 0.4)
        contact.zero_()
        for step in range(8, 22):
            env.episode_length_buf.fill_(step)
            if step == 9:
                command[1, 0] = 0.3
            term(env, "twist", "feet", 0.4)
        command[1, 0] = 0.6
        contact[:] = torch.tensor([0, 0, 1, 1])
        env.episode_length_buf.fill_(30)
        torch.testing.assert_close(term(env, "twist", "feet", 0.4), torch.zeros(2))

    def test_cad_frames_are_aligned_and_centered_on_hip_pairs(self):
        import mujoco
        from mjlab_microduck.robot.quadruped_waist_constants import get_waist_spec, WAIST_HOME_JOINT_POS
        model = get_waist_spec().compile()
        data = mujoco.MjData(model)
        for name, value in WAIST_HOME_JOINT_POS.items():
            data.qpos[model.jnt_qposadr[model.joint(name).id]] = value
        mujoco.mj_forward(model, data)
        for frame, right, left in (("front_gait_frame", "r1_3_foot", "L1_3_foot"),
                                   ("rear_gait_frame", "r2_3_foot", "L2_3_foot")):
            fid = model.site(frame).id
            rotation = torch.tensor(data.site_xmat[fid].reshape(3, 3))
            torch.testing.assert_close(rotation, torch.eye(3, dtype=torch.float64), atol=1e-6, rtol=0)
            rpos = rotation.T @ torch.tensor(data.site_xpos[model.site(right).id] - data.site_xpos[fid])
            lpos = rotation.T @ torch.tensor(data.site_xpos[model.site(left).id] - data.site_xpos[fid])
            self.assertLess(rpos[1].item(), -0.019)
            self.assertGreater(lpos[1].item(), 0.019)
            self.assertAlmostEqual(rpos[0].item(), lpos[0].item(), places=5)
            self.assertAlmostEqual(rpos[2].item(), lpos[2].item(), places=5)


if __name__ == "__main__":
    unittest.main()
