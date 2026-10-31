"""Source-CAD quadruped with XgoDuck Walk's HLS1910 hardware model.

The CAD files are deliberately unchanged. This task is run from the source
checkout (the top-level MJCF directory is not part of the Python wheel).
"""

from copy import deepcopy
from pathlib import Path

import mujoco
import numpy as np
from mjlab.entity import EntityCfg

from mjlab_microduck.robot.xgoduck_constants import XGODUCK_WALK_ROBOT_CFG

QUADRUPED_XML = (
    Path(__file__).resolve().parents[3] / "MJCF"
    / "机械狗总装1.6.SLDASM_mjcf" / "机械狗总装1.6.SLDASM"
    / "机械狗总装1.6.SLDASM.xml"
)
# CAD order: front right, rear right, front left, rear left.
FOOT_BODIES = ("r1_3", "r2_3", "L1_3", "L2_3")
FOOT_SITES = tuple(f"{body}_foot" for body in FOOT_BODIES)
FOOT_GEOMS = tuple(f"{body}_sole_collision" for body in FOOT_BODIES)
HOME_JOINT_POS = {
    "joint1": 0.0, "joint2": -0.8, "joint3": 1.6,
    "joint4": 0.0, "joint5": -0.8, "joint6": 1.6,
    "joint7": 0.0, "joint8": 0.8, "joint9": 1.6,
    "joint10": 0.0, "joint11": 0.8, "joint12": 1.6,
}
# Measured from transformed CAD mesh vertices at HOME: 2 mm sole clearance.
STAND_HEIGHT = 0.13987165
SOLE_START_X = 0.065  # Distal pad only; knee is at local x=0.
FOOT_SITE_POS = {
    body: (0.07977944, (-1 if body.startswith("L") else 1) * 0.02594737, 0.016)
    for body in FOOT_BODIES
}


def _split_distal_collisions(spec: mujoco.MjSpec) -> None:
    """Separate CAD soles from shanks without changing visual meshes/inertia."""
    model = spec.compile()
    # Capture handles before inserting geoms: compiled name IDs become stale as
    # insertion into an earlier body shifts the following geom indices.
    shanks = {geom.name: geom for geom in spec.geoms}
    for body_name, sole_name in zip(FOOT_BODIES, FOOT_GEOMS, strict=True):
        gid = model.geom(f"{body_name}_collision").id
        mid = model.geom_dataid[gid]
        start, count = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, model.geom_quat[gid])
        vertices = model.mesh_vert[start:start + count] @ rotation.reshape(3, 3).T
        vertices += model.geom_pos[gid]
        start, count = model.mesh_faceadr[mid], model.mesh_facenum[mid]
        faces = model.mesh_face[start:start + count]
        edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
        a, b = vertices[edges[:, 0]], vertices[edges[:, 1]]
        crosses = (a[:, 0] < SOLE_START_X) != (b[:, 0] < SOLE_START_X)
        a, b = a[crosses], b[crosses]
        ratio = (SOLE_START_X - a[:, 0]) / (b[:, 0] - a[:, 0])
        cut = a + ratio[:, None] * (b - a)
        for part, mask in (("shank", vertices[:, 0] <= SOLE_START_X),
                           ("sole", vertices[:, 0] >= SOLE_START_X)):
            points = np.unique(np.concatenate((vertices[mask], cut)), axis=0)
            spec.add_mesh(name=f"{body_name}_{part}", uservert=points.ravel().tolist())
        geom = shanks[f"{body_name}_collision"]
        geom.meshname = f"{body_name}_shank"
        spec.body(body_name).add_geom(
            name=sole_name, type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname=f"{body_name}_sole", group=3,
            contype=geom.contype, conaffinity=geom.conaffinity,
            condim=geom.condim, friction=geom.friction.copy(), rgba=geom.rgba.copy(),
        )


def get_spec(
    xml_path: Path = QUADRUPED_XML,
    home_joint_pos: dict[str, float] = HOME_JOINT_POS,
    foot_site_pos: dict[str, tuple[float, float, float]] = FOOT_SITE_POS,
) -> mujoco.MjSpec:
    """Add a free root and named contacts without changing CAD inertia/limits."""
    spec = mujoco.MjSpec.from_file(str(xml_path))
    spec.body("base_link").add_freejoint(name="root")
    spec.add_sensor(
        name="root_angmom", type=mujoco.mjtSensor.mjSENS_SUBTREEANGMOM,
        objtype=mujoco.mjtObj.mjOBJ_BODY, objname="base_link",
    )
    # Copy the biped's bare-MuJoCo companion servo settings too. BAM replaces
    # these with voltage/back-EMF motors and its own force limit at env creation.
    reference = XGODUCK_WALK_ROBOT_CFG.spec_fn()
    servo = reference.actuators[0]
    joint_reference = reference.joint("left_hip_yaw")
    for joint in spec.joints:
        if joint.name in home_joint_pos:
            joint.armature = joint_reference.armature
            joint.damping = joint_reference.damping.copy()
            joint.frictionloss = joint_reference.frictionloss
    for actuator in spec.actuators:
        actuator.gainprm = servo.gainprm.copy()
        actuator.biasprm = servo.biasprm.copy()
        actuator.forcelimited = servo.forcelimited
        actuator.forcerange = servo.forcerange.copy()
        actuator.ctrllimited = servo.ctrllimited
        actuator.ctrlrange = servo.ctrlrange.copy()
    for body in spec.bodies:
        for geom in body.geoms:
            if geom.group == 3:
                geom.name = f"{body.name}_collision"
                # Floor contacts only, like XgoDuck Walk. CAD shells overlap at
                # joints, so enabling all self contacts creates false impulses.
                geom.contype, geom.conaffinity = 1, 0
                geom.condim = 3
                geom.friction = [1.0, 0.005, 0.0001]
            else:
                geom.contype = geom.conaffinity = 0
    for body_name, site_name in zip(FOOT_BODIES, FOOT_SITES, strict=True):
        # Distal sole point measured from the source mesh at HOME, centered
        # across the 12 mm pad width. Left lower-link frames mirror local y.
        spec.body(body_name).add_site(
            name=site_name, pos=foot_site_pos[body_name],
            size=[0.002, 0.002, 0.002], rgba=[0, 1, 0, 1],
        )
    _split_distal_collisions(spec)
    return spec


QUADRUPED_ROBOT_CFG = EntityCfg(
    spec_fn=get_spec,
    init_state=EntityCfg.InitialStateCfg(
        pos=(0.0, 0.0, STAND_HEIGHT),
        joint_pos={f"^{name}$": value for name, value in HOME_JOINT_POS.items()},
        joint_vel={".*": 0.0},
    ),
    articulation=deepcopy(XGODUCK_WALK_ROBOT_CFG.articulation),
)
QUADRUPED_ROBOT_CFG.articulation.actuators[0].target_names_expr = (r"joint[0-9]+",)
