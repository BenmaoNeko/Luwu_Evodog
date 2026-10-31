"""Quadruped integration: CAD kinematics, shared hardware, isolated task config."""

from pathlib import Path
import unittest


class QuadrupedWalkTests(unittest.TestCase):
    def test_entity_initial_joint_patterns_do_not_alias_joint1_and_joint11(self):
        from mjlab.utils.string import resolve_expr
        from mjlab_microduck.robot.quadruped_constants import QUADRUPED_ROBOT_CFG, HOME_JOINT_POS
        names = tuple(HOME_JOINT_POS)
        actual = resolve_expr(QUADRUPED_ROBOT_CFG.init_state.joint_pos, names, 0.0)
        self.assertEqual(actual, tuple(HOME_JOINT_POS.values()))

    def test_distal_collision_separates_knee_from_sole(self):
        import mujoco
        import numpy as np
        from mjlab_microduck.robot.quadruped_constants import get_spec, FOOT_GEOMS, FOOT_BODIES
        model = get_spec().compile()
        for body, foot in zip(FOOT_BODIES, FOOT_GEOMS):
            self.assertNotEqual(foot, f"{body}_collision")
            self.assertEqual(model.geom(foot).bodyid, model.body(body).id)
            self.assertEqual(model.geom(f"{body}_collision").bodyid, model.body(body).id)
            for name, is_sole in ((foot, True), (f"{body}_collision", False)):
                gid = model.geom(name).id
                mid = model.geom_dataid[gid]
                start, count = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
                rotation = np.empty(9)
                mujoco.mju_quat2Mat(rotation, model.geom_quat[gid])
                vertices = model.mesh_vert[start:start + count] @ rotation.reshape(3, 3).T + model.geom_pos[gid]
                if is_sole:
                    self.assertGreaterEqual(vertices[:, 0].min(), 0.065 - 1e-6)
                else:
                    self.assertLessEqual(vertices[:, 0].max(), 0.065 + 1e-6)

    def test_rear_knee_ground_contact_is_not_foot_contact(self):
        import mujoco
        import numpy as np
        from mjlab_microduck.robot.quadruped_constants import get_spec, HOME_JOINT_POS, FOOT_GEOMS
        spec = get_spec()
        spec.worldbody.add_geom(name="terrain", type=mujoco.mjtGeom.mjGEOM_PLANE,
                                size=[1, 1, 0.1], contype=1, conaffinity=1)
        model = spec.compile()
        data = mujoco.MjData(model)
        for name, value in HOME_JOINT_POS.items():
            data.qpos[model.joint(name).qposadr[0]] = value
        data.qpos[model.joint("joint5").qposadr[0]] = -1.2
        data.qpos[model.joint("joint6").qposadr[0]] = 0.2
        mujoco.mj_forward(model, data)
        gid = model.geom("r2_3_collision").id
        mid = model.geom_dataid[gid]
        start, count = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
        vertices = model.mesh_vert[start:start + count] @ data.geom_xmat[gid].reshape(3, 3).T + data.geom_xpos[gid]
        data.qpos[2] -= vertices[:, 2].min() + 0.0001
        mujoco.mj_forward(model, data)
        touched = {model.geom(int(g)).name for contact in data.contact for g in contact.geom}
        self.assertIn("r2_3_collision", touched)
        self.assertNotIn(FOOT_GEOMS[1], touched)


    def test_robot_adapter_exists(self):
        self.assertTrue((Path(__file__).parents[1] / "src/mjlab_microduck/robot/quadruped_constants.py").is_file())

    def test_floating_model_home_and_cad_limits(self):
        import mujoco
        import numpy as np
        from mjlab_microduck.robot.quadruped_constants import (
            QUADRUPED_XML, HOME_JOINT_POS, STAND_HEIGHT, FOOT_SITES, get_spec,
        )
        original_bytes = QUADRUPED_XML.read_bytes()
        source = mujoco.MjModel.from_xml_path(str(QUADRUPED_XML))
        model = get_spec().compile()
        self.assertEqual((model.nq, model.nv, model.nu), (19, 18, 12))
        self.assertEqual(model.jnt_type[0], mujoco.mjtJoint.mjJNT_FREE)
        np.testing.assert_allclose(model.jnt_range[1:], source.jnt_range)
        np.testing.assert_allclose(model.body_mass, source.body_mass)
        data = mujoco.MjData(model)
        data.qpos[2] = STAND_HEIGHT
        for name, value in HOME_JOINT_POS.items():
            joint = model.joint(name)
            data.qpos[joint.qposadr[0]] = value
            self.assertTrue(joint.range[0] < value < joint.range[1])
        mujoco.mj_forward(model, data)
        feet = np.array([data.site_xpos[model.site(n).id] for n in FOOT_SITES])
        np.testing.assert_allclose(feet[:, 2], 0.002, atol=2e-5)
        np.testing.assert_allclose(feet[:2, 0], feet[2:, 0], atol=1e-6)
        np.testing.assert_allclose(feet[:2, 1], -feet[2:, 1], atol=1e-6)
        self.assertTrue(np.isfinite(data.qacc).all())
        self.assertEqual(QUADRUPED_XML.read_bytes(), original_bytes)

    def test_actuator_hardware_is_copied_without_biped_mutation(self):
        from dataclasses import asdict
        from mjlab_microduck.robot.quadruped_constants import QUADRUPED_ROBOT_CFG
        from mjlab_microduck.robot.xgoduck_constants import XGODUCK_WALK_ROBOT_CFG
        quadruped = QUADRUPED_ROBOT_CFG.articulation.actuators[0]
        biped = XGODUCK_WALK_ROBOT_CFG.articulation.actuators[0]
        self.assertIsNot(quadruped, biped)
        expected = asdict(biped)
        expected["target_names_expr"] = (r"joint[0-9]+",)
        self.assertEqual(asdict(quadruped), expected)

    def test_task_registration_curriculum_and_play(self):
        import mjlab_microduck.tasks
        from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
        task = "Mjlab-Velocity-Flat-Quadruped"
        cfg = load_env_cfg(task)
        play = load_env_cfg(task, play=True)
        self.assertEqual(cfg.scene.terrain.terrain_type, "plane")
        self.assertEqual(set(cfg.commands), {"twist"})
        self.assertNotIn("head_pose", cfg.observations["actor"].terms)
        self.assertNotIn("base_lin_vel", cfg.observations["actor"].terms)
        self.assertEqual(cfg.sim.mujoco.timestep * cfg.decimation, 0.02)
        self.assertEqual(cfg.events["reset_base"].params["pose_range"]["z"], (0.0, 0.0))
        self.assertFalse(cfg.curriculum)
        self.assertEqual(cfg.commands["twist"].ranges.lin_vel_x, (-0.2, 0.3))
        self.assertIn("phase", cfg.observations["actor"].terms)
        self.assertIn("phase", cfg.observations["critic"].terms)
        self.assertEqual(cfg.rewards["foot_gait"].params["offset"], [0.5, 0.0, 0.0, 0.5])
        self.assertNotIn("prolonged_foot_air_time", cfg.rewards)
        self.assertFalse(play.curriculum)
        self.assertFalse(play.observations["actor"].enable_corruption)
        self.assertEqual(play.commands["twist"].ranges.lin_vel_x, (-0.2, 0.3))
        self.assertEqual(load_rl_cfg(task).experiment_name, "quadruped_velocity")

    def test_luwu_gait_requires_alternating_diagonal_contacts(self):
        import torch
        from types import SimpleNamespace as NS
        from mjlab_microduck.tasks.luwu_quadruped_mdp import feet_gait, phase
        commands = torch.tensor([[0.2, 0.0, 0.0]])
        sensor = NS(data=NS(current_contact_time=torch.tensor([[0.0, 0.1, 0.1, 0.0]])))
        env = NS(scene={"feet": sensor}, device="cpu", num_envs=1, step_dt=0.02,
                 episode_length_buf=torch.tensor([2]),
                 command_manager=NS(get_command=lambda _: commands))
        params = dict(period=0.4, offset=[0.5, 0.0, 0.0, 0.5], threshold=0.56,
                      command_threshold=0.1, command_name="twist", sensor_name="feet")
        self.assertEqual(feet_gait(env, **params).item(), 1.0)
        env.episode_length_buf[:] = 12
        self.assertEqual(feet_gait(env, **params).item(), 0.0)
        sensor.data.current_contact_time = torch.tensor([[0.1, 0.0, 0.0, 0.1]])
        self.assertEqual(feet_gait(env, **params).item(), 1.0)
        commands.zero_()
        self.assertEqual(feet_gait(env, **params).item(), 0.0)
        self.assertTrue((phase(env, 0.4, "twist") == 0).all())


if __name__ == "__main__":
    unittest.main()
