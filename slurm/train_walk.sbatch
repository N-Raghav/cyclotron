#!/bin/bash
#SBATCH --job-name=asimov_walk_amp
#SBATCH --account=gli7
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --constraint=L40S|RTX6000B
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/rnallaperumal/cyclotron/slurm/logs/%x_%j.out

# Low-level walking policy for the hierarchical soccer task (stock AMP baseline).
cd /home/rnallaperumal/cyclotron
source .venv/bin/activate
hostname; nvidia-smi --query-gpu=name --format=csv,noheader
./cyclotron.sh --train --task Asimov1-Velocity-AMP-v0 --num_envs 4096 --headless "$@"
