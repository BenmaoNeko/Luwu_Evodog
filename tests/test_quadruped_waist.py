"""Waist CAD adaptation, shared hardware, and real reset regressions (remote)."""
from dataclasses import asdict
from pathlib import Path
import unittest


class WaistQuadrupedTests(unittest.TestCase):
    def test_adapter_exists(self):
        self.assertTrue((Path(__file__).parents[1] / "src/mjlab_microduck/robot/quadruped_waist_constants.py").is_file())

    def test_shared_hardware_and_original_limits(self):
        import mujoco
        import numpy as np
        from mjlab_microduck.robot.quadruped_constants import QUADRUPED_ROBOT_CFG
        from mjlab_microduck.robot.quadruped_waist_constants import (
            WAIST_QUADRUPED_XML, WAIST_QUADRUPED_ROBOT_CFG, get_waist_spec,
        )
        self.assertEqual(asdict(WAIST_QUADRUPED_ROBOT_CFG.articulation),
                         asdict(QUADRUPED_ROBOT_CFG.articulation))
        old = QUADRUPED_ROBOT_CFG.spec_fn().compile()
        model = get_waist_spec().compile()
        source = mujoco.MjModel.from_xml_path(str(WAIST_QUADRUPED_XML))
        self.assertEqual((model.nq, model.nv, model.nu), (20, 19, 13))
        np.testing.assert_allclose(model.body_mass, source.body_mass)
        np.testing.assert_allclose(model.body_inertia, source.body_inertia)
        for i in range(1, model.njnt):
            name = model.joint(i).name
            np.testing.assert_allclose(model.joint(name).range, source.joint(name).range)
        for i in range(model.nu):
            for field in ("gainprm", "biasprm", "ctrlrange", "forcerange", "ctrllimited", "forcelimited"):
                np.testing.assert_allclose(getattr(model.actuator(i), field), getattr(old.actuator(0), field))
        for name in (f"joint{i}" for i in range(1, 14)):
            dof = model.joint(name).dofadr[0]
            ref = old.joint("joint1").dofadr[0]
            for field in ("dof_armature", "dof_damping", "dof_frictionloss"):
                np.testing.assert_allclose(getattr(model, field)[dof], getattr(old, field)[ref])

    def test_exact_defaults_and_four_soles(self):
        import mujoco
        import numpy as np
        from mjlab.utils.string import resolve_expr
        from mjlab_microduck.robot.quadruped_constants import FOOT_SITES, FOOT_GEOMS, FOOT_BODIES
        from mjlab_microduck.robot.quadruped_waist_constants import (
            WAIST_QUADRUPED_ROBOT_CFG, WAIST_HOME_JOINT_POS, get_waist_spec,
        )
        cfg = WAIST_QUADRUPED_ROBOT_CFG
        model = get_waist_spec().compile()
        data = mujoco.MjData(model)
        names = tuple(model.joint(i).name for i in range(1, model.njnt))
        resolved = resolve_expr(cfg.init_state.joint_pos, names, 0.0)
        self.assertEqual(resolved, tuple(WAIST_HOME_JOINT_POS[n] for n in names))
        self.assertEqual(WAIST_HOME_JOINT_POS["joint13"], 0)
        data.qpos[:3] = cfg.init_state.pos
        for name, value in zip(names, resolved):
            data.qpos[model.joint(name).qposadr[0]] = value
        mujoco.mj_forward(model, data)
        np.testing.assert_allclose([data.site_xpos[model.site(n).id, 2] for n in FOOT_SITES], .002, atol=2e-6)
        for body, foot in zip(FOOT_BODIES, FOOT_GEOMS):
            self.assertNotEqual(model.geom(foot).id, model.geom(f"{body}_collision").id)

    def test_task_reset_matches_no_waist(self):
        import mjlab_microduck.tasks  # noqa: F401 -- registers tasks
        from mjlab.tasks.registry import load_env_cfg
        old = load_env_cfg("Mjlab-Velocity-Flat-Quadruped")
        cfg = load_env_cfg("Mjlab-Velocity-Flat-Quadruped-Waist")
        for name in ("reset_base", "reset_robot_joints"):
            self.assertEqual(asdict(cfg.events[name]), asdict(old.events[name]))
        self.assertEqual(cfg.actions["joint_pos"].scale, old.actions["joint_pos"].scale)
        self.assertEqual(cfg.scene.terrain.terrain_type, "plane")
        self.assertEqual(cfg.curriculum, old.curriculum)

    def test_real_reset_no_clamping_or_floating_leg(self):
        import numpy as np
        import torch
        import mjlab_microduck.tasks  # noqa: F401 -- registers tasks
        from mjlab.envs import ManagerBasedRlEnv
        from mjlab.tasks.registry import load_env_cfg
        from mjlab_microduck.robot.quadruped_waist_constants import WAIST_HOME_JOINT_POS
        cfg = load_env_cfg("Mjlab-Velocity-Flat-Quadruped-Waist")
        cfg.scene.num_envs = 1
        env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
        try:
            robot = env.scene["robot"]
            foot_ids, _ = robot.find_sites(".*_foot")
            for seed in (42, 43, 44):
                env.reset(seed=seed)
                expected = [WAIST_HOME_JOINT_POS[n] for n in robot.joint_names]
                np.testing.assert_allclose(robot.data.joint_pos[0].cpu(), expected, atol=1e-6)
                np.testing.assert_allclose(robot.data.site_pos_w[0, foot_ids, 2].cpu(), .002, atol=2e-6)
                np.testing.assert_allclose(robot.data.projected_gravity_b[0].cpu(), [0, 0, -1], atol=1e-6)
                np.testing.assert_allclose(robot.data.joint_vel[0].cpu(), 0, atol=1e-6)
            print("WAIST_RESET", "joints", dict(zip(robot.joint_names, robot.data.joint_pos[0].cpu().tolist())),
                  "sole_z", robot.data.site_pos_w[0, foot_ids, 2].cpu().tolist(), flush=True)
            for _ in range(20):
                env.step(torch.zeros((1, 13), device=env.device))
            feet = env.scene["feet_ground_contact"].data.found
            nonfoot = env.scene["nonfoot_ground_contact"].data.found
            print("WAIST_SETTLE", "feet", feet.cpu().tolist(), "nonfoot", nonfoot.cpu().tolist(), flush=True)
            self.assertTrue((feet > 0).all().item())
            self.assertFalse((nonfoot > 0).any().item())
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
