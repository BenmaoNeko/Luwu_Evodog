"""13-joint waist CAD, using the corrected quadruped's unchanged hardware."""
from copy import deepcopy
from pathlib import Path

import mujoco

from mjlab_microduck.robot.quadruped_constants import (
    FOOT_BODIES, HOME_JOINT_POS, QUADRUPED_ROBOT_CFG, STAND_HEIGHT, get_spec,
)

WAIST_QUADRUPED_XML = (
    Path(__file__).resolve().parents[3] / "MJCF"
    / "机械狗总装（含腰）1.6URDF_mjcf (1)" / "机械狗总装（含腰）1.6URDF"
    / "机械狗总装（含腰）1.6URDF.xml"
)
# This CAD reverses the left-front thigh axis relative to the no-waist CAD.
WAIST_HOME_JOINT_POS = {**HOME_JOINT_POS, "joint8": -0.8, "joint13": 0.0}
WAIST_FOOT_SITE_POS = {
    body: (0.07977944, -0.02594737, 0.016 if body.startswith("L") else -0.016)
    for body in FOOT_BODIES
}


def get_waist_spec():
    spec = get_spec(WAIST_QUADRUPED_XML, WAIST_HOME_JOINT_POS, WAIST_FOOT_SITE_POS)
    # Base CAD bounds: x=[-0.05747846, 0.04], top z=0.001 m.
    battery = spec.body("base_link").add_body(
        name="battery", pos=[-0.00873923, 0.0, 0.010],
        mass=0.099, inertia=[0.00001277925, 0.00003752925, 0.0000449625],
        explicitinertial=True,
    )
    battery.add_geom(
        name="battery_collision", type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.0325, 0.0175, 0.009], rgba=[0.15, 0.15, 0.18, 1.0],
        group=3, contype=1, conaffinity=0, condim=3,
        friction=[1.0, 0.005, 0.0001],
    )
    for torso, hips, name in (
        ("base_link", ("r1", "L1"), "front_gait_frame"),
        ("w2", ("r2", "L2"), "rear_gait_frame"),
    ):
        body = spec.body(torso)
        midpoint = (spec.body(hips[0]).pos + spec.body(hips[1]).pos) / 2
        # Undo the rear CAD's fixed 90-degree roll; retain the live waist motion.
        rotation = body.quat.copy()
        rotation[1:] *= -1
        body.add_site(name=name, pos=midpoint, quat=rotation, size=[0.001] * 3,
                      rgba=[0, 0, 0, 0], group=5)
    return spec


WAIST_QUADRUPED_ROBOT_CFG = deepcopy(QUADRUPED_ROBOT_CFG)
WAIST_QUADRUPED_ROBOT_CFG.spec_fn = get_waist_spec
WAIST_QUADRUPED_ROBOT_CFG.init_state.pos = (0.0, 0.0, STAND_HEIGHT)
WAIST_QUADRUPED_ROBOT_CFG.init_state.joint_pos = {
    f"^{name}$": value for name, value in WAIST_HOME_JOINT_POS.items()
}
