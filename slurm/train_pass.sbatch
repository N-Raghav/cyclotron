#!/bin/bash
#SBATCH --job-name=asimov_pass
#SBATCH --account=gli7
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --constraint=L40S|RTX6000B|A100
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/rnallaperumal/cyclotron/slurm/logs/%x_%j.out

# Inside-foot pass from scratch: PPO + AMP (pass clip), reference-state init, 5 cm ball.
cd /home/rnallaperumal/cyclotron
source .venv/bin/activate
hostname; nvidia-smi --query-gpu=name --format=csv,noheader
./cyclotron.sh --train --task Asimov1-Pass-v0 --num_envs 4096 --headless "$@"
