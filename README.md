# XgoDuck RL

Reinforcement-learning environments for training XgoDuck, a small biped whose geometry and inertia differ from MicroDuck. Policies are trained with PPO on [mjlab](https://github.com/mujocolab/mjlab) (MuJoCo Warp) and exported to ONNX.

## Acknowledgments

This repository is a downstream project based on [microduck_rl](https://github.com/pollen-robotics/microduck_rl) by [Pollen Robotics](https://github.com/pollen-robotics). Training runs on [mjlab](https://github.com/mujocolab/mjlab). Joint actuation uses [BAM](https://github.com/Rhoban/bam) (Better Actuator Models) from Rhoban.

Thank you to the authors of those projects.

## What is different

The task code follows the MicroDuck recipes, but the robot does not:

- The MJCF and meshes use the XgoDuck model (`src/mjlab_microduck/robot/xgoduck/`).
- Actuators are the HLS1910 BAM model in `src/mjlab_microduck/robot/xgoduck/params/1910_m6.json`, not the MicroDuck XL330.

## Install

You need a CUDA GPU, Python 3.12, and [uv](https://docs.astral.sh/uv/).

```bash
cd xgoduck_rl
uv sync
```

On ARM machines (DGX Spark / GB10, Jetson), the first `uv sync` downloads a large CUDA wheel and uv's default 30 s HTTP timeout can abort it. Set `UV_HTTP_TIMEOUT=600` for that first sync.

## Train

From the repository root:

```bash
uv run train Mjlab-Velocity-Flat-XgoDuck --env.scene.num-envs 4096
uv run train Mjlab-Velocity-Swizzle-XgoDuck --env.scene.num-envs 4096
```

`uv run list-envs` prints every registered task. XgoDuck checkpoints and TensorBoard logs go under `logs/rsl_rl/<experiment>/`.

| Task | Terrain | Experiment directory |
|---|---|---|
| `Mjlab-Velocity-Flat-XgoDuck` | flat | `logs/rsl_rl/xgoduck_velocity/` |
| `Mjlab-Velocity-Rough-XgoDuck` | rough | `logs/rsl_rl/xgoduck_velocity/` |
| `Mjlab-StandUp-Flat-XgoDuck` | flat | `logs/rsl_rl/xgoduck_standup/` |
| `Mjlab-StandUp-Rough-XgoDuck` | rough | `logs/rsl_rl/xgoduck_standup/` |
| `Mjlab-GroundPick-Flat-XgoDuck` | flat | `logs/rsl_rl/xgoduck_ground_pick/` |
| `Mjlab-GroundPick-Rough-XgoDuck` | rough | `logs/rsl_rl/xgoduck_ground_pick/` |
| `Mjlab-SitStand-Flat-XgoDuck` | flat | `logs/rsl_rl/xgoduck_sitstand/` |
| `Mjlab-SitStand-Rough-XgoDuck` | rough | `logs/rsl_rl/xgoduck_sitstand/` |
| `Mjlab-BallKick-Flat-XgoDuck` | flat | `logs/rsl_rl/xgoduck_ball_kick_right/` |
| `Mjlab-BallKick-Left-Flat-XgoDuck` | flat | `logs/rsl_rl/xgoduck_ball_kick_left/` |
| `Mjlab-Roulade-Flat-XgoDuck` | flat | `logs/rsl_rl/xgoduck_roulade/` |
| `Mjlab-Roulade-Angle-Flat-XgoDuck` | flat | `logs/rsl_rl/xgoduck_roulade_angle/` |
| `Mjlab-Velocity-Swizzle-XgoDuck` | flat | `logs/rsl_rl/xgoduck_velocity_swizzle/` |

Both Roulade tasks start standing and run for seven seconds, including standing
after the roll. The baseline retains the original observations and deployment
interface. The Angle variant replaces one of the last six zero-padding values
with signed cumulative rotation in radians; the other five stay zero. Training
adds uniform ±30° angle noise. It uses the same rewards as the baseline, with
no extra 360° angle-tracking reward. Its runtime must accumulate the sagittal
angle from IMU orientation (see `mdp._roulade_pitch_angle`), starting at zero
on skill entry. Both actors have 61 inputs.

## Play

### Quadruped CAD walking task

`Mjlab-Velocity-Flat-Quadruped` loads the original model from
`MJCF/机械狗总装1.6.SLDASM_mjcf/机械狗总装1.6.SLDASM/` in this source checkout.
The adapter adds a floating base and four foot sites, preserves CAD inertia and
joint limits, and copies the XgoDuck Walk HLS1910 BAM actuator configuration,
including voltage limits and 3–6 physics-step command delay. No biped task is changed.

```bash
uv run train Mjlab-Velocity-Flat-Quadruped --env.scene.num-envs 4096 --agent.max-iterations 3000
uv run play Mjlab-Velocity-Flat-Quadruped
```

The current recipe comes from `luwu_mjlab`'s `lite3_flat_walk_env_cfg`.
Relevant reward, phase observation and illegal-contact functions are vendored
unchanged in `tasks/luwu_quadruped_mdp.py` (Apache-2.0). It uses a 0.4 s gait
period, duty threshold 0.56, action scale 0.25, velocity tracking weights 4.0
and standard deviations 0.15. Commands are sampled over -0.2–0.3 m/s forward,
±0.15 m/s lateral and ±0.8 rad/s yaw from the start, with no velocity curriculum
or explicit standing fraction, matching Lite3 flat walk.

Joint-specific posture tolerances are mapped to the CAD hip/thigh/shank names.
The source XML traversal is FL,FR,RR,RL; its offsets [0,0.5,0,0.5] map to
[0.5,0,0,0.5] for our FR,RR,FL,RL contact order (diagonal pairs).
This adds two phase inputs to actor and critic, requiring fresh training;
old checkpoints have incompatible observation dimensions. The BAM timestep,
delays and hardware settings stay those of XgoDuck Walk (5 ms, decimation 4).
The original CAD/height/reset guards and startup randomization are retained.
Checkpoints live under `logs/rsl_rl/quadruped_velocity/` with save interval 100.

For the designated remote server, `bash scripts/train_quadruped_remote.sh` runs
4096 environments for 3000 iterations. Set `QUADRUPED_PYTHON` to override the
existing server interpreter; the script uses this checkout's code and vendored
BAM through `PYTHONPATH` without modifying the other project's environment.

`play` with no checkpoint flag loads the newest `model_*.pt` in that task's experiment directory:

```bash
uv run play Mjlab-Velocity-Flat-XgoDuck
uv run play Mjlab-Velocity-Swizzle-XgoDuck
```

Pass `--checkpoint-file path/to/model_XXXX.pt` to choose a specific checkpoint.

## ONNX

Export bakes observation normalization into the graph. Run it from the repository root and point it at a trained checkpoint:

```bash
uv run python scripts/export.py Mjlab-Velocity-Flat-XgoDuck \
    --checkpoint-file logs/rsl_rl/xgoduck_velocity/<run>/model_XXXX.pt \
    --onnx-file logs/rsl_rl/xgoduck_velocity/<run>/<run>.onnx
```

If you omit `--onnx-file`, the file is written to `output.onnx` in the current directory. `*.onnx` is gitignored. Keep the export next to the run that produced it, for example `logs/rsl_rl/xgoduck_velocity/<run>/<run>.onnx`.

## License

The code is licensed under Apache 2.0. See [LICENSE](LICENSE).


## Handoff contents

This checkout includes training/inference code, tests, robot MJCF/mesh resources,
and the vendored BAM dependency. Local experiment notes, debug/preview utilities,
logs, caches, archives and the unrelated `luwu_mjlab-main` source backup are omitted.
The copied Luwu reward implementation remains under `src/`; its upstream license
is retained under `vendor/licenses/`. No trained PT/ONNX checkpoints were present
in the local project at handoff time. Supply your own checkpoint for playback.
