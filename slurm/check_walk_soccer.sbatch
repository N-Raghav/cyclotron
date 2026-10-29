#!/bin/bash
#SBATCH --job-name=asimov_walk_check
#SBATCH --account=gli7
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --constraint=L40S|RTX6000B
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/home/rnallaperumal/cyclotron/slurm/logs/%x_%j.out

# Export the soccer-feet walker fine-tune, test every command on the soccer robot, and render GIFs.
cd /home/rnallaperumal/cyclotron
source .venv/bin/activate
RUN=2026-10-28_22-23-17_amp_soccer
DIR=logs/rsl_rl/asimov_velocity_amp_soccer/$RUN
OUT=outputs/walk_check/soccer_finetune
mkdir -p $OUT
echo "=== export"
timeout 900 ./cyclotron.sh --export --task Asimov1-Velocity-AMP-Soccer-v0 --load_run $RUN --checkpoint model_11998.pt \
    > $OUT/export.log 2>&1; echo "exit=$?"; ls $DIR/exported || { tail -30 $OUT/export.log; exit 1; }
echo "=== walk test"
timeout 900 python -u scripts/soccer/record_walk.py --policy $DIR/exported/policy.pt --robot soccer --out_dir $OUT \
    > $OUT/record.log 2>&1; echo "exit=$?"
grep -E '^\[walk\]|Traceback|^\w*Error' $OUT/record.log
echo "=== replay"
for f in $OUT/soccer_*.npz; do
    MUJOCO_GL=egl python scripts/soccer/replay_mujoco.py $f --width 400 --height 300 2>&1 | grep -E 'wrote|Error'
done
