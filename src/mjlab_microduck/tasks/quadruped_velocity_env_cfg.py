"""CAD quadruped using luwu_mjlab's Lite3 flat walking reward recipe."""

from copy import deepcopy
import math

from mjlab.envs import mdp as env_mdp
from mjlab.managers import EventTermCfg, ObservationTermCfg, RewardTermCfg, TerminationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import (
    ContactMatch, ContactSensorCfg, ObjRef, RingPatternCfg, TerrainHeightSensorCfg,
)
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

from mjlab_microduck.robot.quadruped_constants import (
    FOOT_GEOMS, FOOT_SITES, QUADRUPED_ROBOT_CFG,
)
from mjlab_microduck.tasks.mdp import expand_bam_friction_fields, robot_state_is_nan
from mjlab_microduck.tasks.microduck_velocity_env_cfg import MicroduckRlCfg
from mjlab_microduck.tasks import luwu_quadruped_mdp as luwu_mdp

def make_quadruped_velocity_env_cfg(play: bool = False):
    cfg = make_velocity_env_cfg()
    cfg.scene.entities = {"robot": deepcopy(QUADRUPED_ROBOT_CFG)}
    cfg.scene.num_envs = 1024
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None
    cfg.scene.sensors = (
        ContactSensorCfg(
            name="nonfoot_ground_contact",
            primary=ContactMatch(
                mode="geom", pattern=".*_collision", entity="robot", exclude=FOOT_GEOMS,
            ),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("found", "force"), reduce="none", num_slots=1, history_length=4,
        ),
        ContactSensorCfg(
            name="feet_ground_contact",
            primary=ContactMatch(mode="geom", pattern="^(" + "|".join(FOOT_GEOMS) + ")$", entity="robot"),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("found", "force"), reduce="netforce", num_slots=1,
            track_air_time=True,
        ),
        TerrainHeightSensorCfg(
            name="foot_height_scan",
            frame=tuple(ObjRef(type="site", name=n, entity="robot") for n in FOOT_SITES),
            pattern=RingPatternCfg.single_ring(radius=0.01, num_samples=2),
            ray_alignment="yaw", max_distance=1.0,
            exclude_parent_body=True, include_geom_groups=(0,), debug_vis=False,
        ),
    )
    cfg.viewer.body_name = "base_link"
    cfg.viewer.distance = 0.7
    cfg.viewer.elevation = -20
    cfg.actions["joint_pos"].scale = 0.25
    # No ground-truth linear velocity in the deployable actor. Critic retains it.
    for group in cfg.observations.values():
        group.terms.pop("height_scan", None)
        group.terms["base_lin_vel"].func = env_mdp.base_lin_vel
        group.terms["base_lin_vel"].params = {}
        group.terms["base_ang_vel"].func = env_mdp.base_ang_vel
        group.terms["base_ang_vel"].params = {}
        group.terms["base_ang_vel"].noise.n_min = -0.06
        group.terms["base_ang_vel"].noise.n_max = 0.06
        group.terms["phase"] = ObservationTermCfg(
            func=luwu_mdp.phase, params={"period": 0.4, "command_name": "twist"},
        )
    cfg.observations["actor"].terms.pop("base_lin_vel")

    command = cfg.commands["twist"]
    command.heading_command = False
    command.ranges.heading = None
    command.rel_heading_envs = 0.0
    command.rel_standing_envs = 0.0
    command.rel_forward_envs = 0.0
    command.ranges.lin_vel_x = (-0.2, 0.3)
    command.ranges.lin_vel_y = (-0.15, 0.15)
    command.ranges.ang_vel_z = (-0.8, 0.8)
    cfg.curriculum.clear()  # Lite3 flat walk samples its full ranges from the start.

    cfg.events["reset_base"].params["pose_range"]["z"] = (0.0, 0.0)
    cfg.events["foot_friction"].params["asset_cfg"].geom_names = FOOT_GEOMS
    cfg.events["foot_friction"].params["ranges"] = (0.4, 1.5)
    cfg.events["base_com"].params["asset_cfg"].body_names = ("base_link",)
    cfg.events["base_com"].params["ranges"] = {i: (-0.002, 0.002) for i in range(3)}
    # No pushes during this initial walking course.
    cfg.events.pop("push_robot")
    cfg.events["expand_bam_friction_fields"] = EventTermCfg(
        func=expand_bam_friction_fields, mode="startup",
    )
    body = SceneEntityCfg("robot", body_names=("base_link",))
    feet = SceneEntityCfg("robot", site_names=FOOT_SITES)
    joints = SceneEntityCfg("robot", joint_names=("joint[0-9]+",))
    # Original Lite3 reward weights and formulas. Match its MODEL traversal
    # order FL,FR,RR,RL (not its FOOT_NAMES constant) to our FR,RR,FL,RL.
    cfg.rewards = {
        "track_linear_velocity": RewardTermCfg(
            func=luwu_mdp.track_linear_velocity, weight=4.0,
            params={"command_name": "twist", "std": 0.15},
        ),
        "track_angular_velocity": RewardTermCfg(
            func=luwu_mdp.track_angular_velocity, weight=4.0,
            params={"command_name": "twist", "std": 0.15},
        ),
        "body_orientation_l2": RewardTermCfg(
            func=luwu_mdp.body_orientation_l2, weight=-3.0, params={"asset_cfg": body},
        ),
        "pose": RewardTermCfg(
            func=luwu_mdp.variable_posture, weight=1.0,
            params={
                "asset_cfg": joints, "command_name": "twist",
                "std_standing": {r"joint(1|4|7|10)": 0.05, r"joint(2|5|8|11)": 0.1, r"joint(3|6|9|12)": 0.15},
                "std_walking": {r"joint(1|4|7|10)": 0.15, r"joint(2|5|8|11)": 0.4, r"joint(3|6|9|12)": 0.4},
                "std_running": {r"joint(1|4|7|10)": 0.25, r"joint(2|5|8|11)": 0.6, r"joint(3|6|9|12)": 0.6},
                "walking_threshold": 0.05, "running_threshold": 0.3,
            },
        ),
        "body_ang_vel": RewardTermCfg(
            func=luwu_mdp.body_angular_velocity_penalty, weight=-0.08, params={"asset_cfg": body},
        ),
        "angular_momentum": RewardTermCfg(
            func=luwu_mdp.angular_momentum_penalty, weight=-0.03,
            params={"sensor_name": "robot/root_angmom"},
        ),
        "is_terminated": RewardTermCfg(func=env_mdp.is_terminated, weight=-200.0),
        "joint_acc_l2": RewardTermCfg(func=env_mdp.joint_acc_l2, weight=-2.5e-7),
        "joint_pos_limits": RewardTermCfg(func=env_mdp.joint_pos_limits, weight=-10.0),
        "action_rate_l2": RewardTermCfg(func=env_mdp.action_rate_l2, weight=-0.15),
        "foot_gait": RewardTermCfg(
            func=luwu_mdp.feet_gait, weight=1.0,
            params={"period": 0.4, "offset": [0.5, 0.0, 0.0, 0.5], "threshold": 0.56,
                    "command_threshold": 0.1, "command_name": "twist", "sensor_name": "feet_ground_contact"},
        ),
        "foot_clearance": RewardTermCfg(
            func=luwu_mdp.feet_clearance, weight=-0.5,
            params={"target_height": 0.02, "command_name": "twist", "command_threshold": 0.1, "asset_cfg": feet},
        ),
        "foot_slip": RewardTermCfg(
            func=luwu_mdp.feet_slip, weight=-0.05,
            params={"sensor_name": "feet_ground_contact", "command_name": "twist", "command_threshold": 0.1, "asset_cfg": feet},
        ),
        "soft_landing": RewardTermCfg(
            func=luwu_mdp.soft_landing, weight=-1e-3,
            params={"sensor_name": "feet_ground_contact", "command_name": "twist", "command_threshold": 0.1},
        ),
        "stand_still": RewardTermCfg(
            func=luwu_mdp.stand_still, weight=-1.0,
            params={"command_name": "twist", "command_threshold": 0.1, "asset_cfg": joints},
        ),
        "nonfoot_contact": RewardTermCfg(
            func=luwu_mdp.illegal_contact, weight=-3.0,
            params={"sensor_name": "nonfoot_ground_contact", "force_threshold": 0.5},
        ),
    }
    cfg.terminations["illegal_contact"] = TerminationTermCfg(
        func=luwu_mdp.illegal_contact,
        params={"sensor_name": "nonfoot_ground_contact", "force_threshold": 10.0},
    )
    cfg.terminations["fell_over"].params["limit_angle"] = math.radians(70)
    cfg.terminations["base_too_low"] = TerminationTermCfg(
        func=env_mdp.root_height_below_minimum, params={"minimum_height": 0.07},
    )
    cfg.terminations["nan_state"] = TerminationTermCfg(
        func=robot_state_is_nan, params={"sensor_names": ("feet_ground_contact",)},
    )
    cfg.sim.nconmax = 64
    if play:
        cfg.scene.num_envs = 1
        cfg.curriculum.clear()
        cfg.observations["actor"].enable_corruption = False
        for name in ("base_com", "encoder_bias", "foot_friction"):
            cfg.events.pop(name)
    return cfg


QuadrupedRlCfg = deepcopy(MicroduckRlCfg)
QuadrupedRlCfg.algorithm.symmetry_cfg = None
QuadrupedRlCfg.logger = "tensorboard"
QuadrupedRlCfg.experiment_name = "quadruped_velocity"
QuadrupedRlCfg.run_name = "quadruped"
QuadrupedRlCfg.upload_model = False
QuadrupedRlCfg.max_iterations = 5000
QuadrupedRlCfg.save_interval = 100
