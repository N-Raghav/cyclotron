#!/bin/bash
#SBATCH --job-name=asimov_pass_clips
#SBATCH --account=gli7
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --constraint=L40S|RTX6000B
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/home/rnallaperumal/cyclotron/slurm/logs/%x_%j.out

# Regenerate the pass reference clips (5 and 7 cm) and copy them into the task's motions folder.
cd /home/rnallaperumal/cyclotron
source .venv/bin/activate
for r in ${RADII:-0.05 0.07}; do
    echo "=== clip r=$r"
    timeout 1500 python -u scripts/soccer/pass_motion.py --ball_radius $r 2>&1 | grep -E '\[motion\]|Error|Traceback'
    cp outputs/soccer_pass/pass_r$r.npz source/cyclotron/cyclotron/tasks/soccer/motions/asimov_1_pass_inside_r$r.npz
done
