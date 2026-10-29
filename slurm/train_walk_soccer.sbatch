#!/bin/bash
#SBATCH --job-name=asimov_walk_soccer
#SBATCH --account=gli7
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --constraint=L40S|RTX6000B
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/home/rnallaperumal/cyclotron/slurm/logs/%x_%j.out

# Fine-tune the AMP walker on the soccer robot (foot boxes, self-collision on), with the style reward limited to the
# forward-walking commands the reference clip covers. Starts from the stock-robot baseline.
cd /home/rnallaperumal/cyclotron
source .venv/bin/activate
hostname; nvidia-smi --query-gpu=name --format=csv,noheader
./cyclotron.sh --train --task Asimov1-Velocity-AMP-Soccer-v0 --num_envs 4096 --headless \
    --checkpoint logs/rsl_rl/asimov_velocity_amp/2026-10-27_00-04-34_amp/model_9999.pt "$@"
