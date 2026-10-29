#!/bin/bash
#SBATCH --job-name=asimov_pass_record
#SBATCH --account=gli7
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --constraint=L40S|RTX6000B|A100
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/home/rnallaperumal/cyclotron/slurm/logs/%x_%j.out

# Export the pass run's latest checkpoint, score one episode in 16 envs, and render GIFs in MuJoCo.
# Usage: [TASK=Asimov1-Pass-R07] [REPLAY_ARGS='--slow 3 --width 400 --height 300'] sbatch slurm/record_pass.sbatch [run] [checkpoint] (defaults: latest run, its latest model_*.pt)
TASK=${TASK:-Asimov1-Pass}
cd /home/rnallaperumal/cyclotron
source .venv/bin/activate
RUN=${1:-$(ls -t logs/rsl_rl/asimov1_pass | head -1)}
DIR=logs/rsl_rl/asimov1_pass/$RUN
CKPT=${2:-$(ls $DIR/model_*.pt | sort -t_ -k2 -n | tail -1)}
IT=$(basename $CKPT .pt | sed 's/model_//')
EXP=$DIR/exported_$IT
OUT=outputs/pass_check/${RUN}_it$IT
echo "=== export $CKPT -> $EXP"
timeout 900 ./cyclotron.sh --export --task $TASK-v0 --load_run $RUN --checkpoint $(basename $CKPT) --output $EXP \
    > $OUT.export.log 2>&1 || mkdir -p outputs/pass_check
echo "exit=$?"; ls $EXP || { tail -30 $OUT.export.log; exit 1; }

echo "=== record"
timeout 900 python -u scripts/soccer/record_pass.py --task $TASK-Play-v0 --policy $EXP/policy.pt --out_dir $OUT > $OUT.record.log 2>&1
echo "exit=$?"; grep -E '^\[pass\]|Traceback|^\w*Error' $OUT.record.log

echo "=== replay"
for f in $OUT/pass_env*.npz; do
    MUJOCO_GL=egl python scripts/soccer/replay_mujoco.py $f --track --distance 2.5 $REPLAY_ARGS 2>&1 | grep -E 'wrote|Error'
done
