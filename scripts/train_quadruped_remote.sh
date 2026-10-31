#!/usr/bin/env bash
# Run from the synced source tree; no editable installation of other projects.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src:$PWD/vendor/bam${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export MUJOCO_GL=egl
export OMP_NUM_THREADS=8
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
export XDG_CACHE_HOME="$PWD/.cache"
PYTHON_BIN="${QUADRUPED_PYTHON:-/root/gpufree-data/wuji-gentle-flip-20261028/.pixi/envs/default/bin/python}"
exec "$PYTHON_BIN" -u -c \
  'import mjlab_microduck.tasks; from mjlab.scripts.train import main; main()' \
  Mjlab-Velocity-Flat-Quadruped \
  --env.scene.num-envs 4096 \
  --agent.max-iterations 3000 \
  --agent.run-name quadruped_reset_contact_scratch_4096_3000 "$@"
