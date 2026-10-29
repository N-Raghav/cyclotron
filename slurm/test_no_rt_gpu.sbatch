#!/bin/bash
#SBATCH --job-name=asimov_nort_test
#SBATCH --account=gli7
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --constraint=A100|H100|H200
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=/home/rnallaperumal/cyclotron/slurm/logs/%x_%j.out

# Can Isaac Sim / Isaac Lab train headless on a GPU without RT cores (A100, H100, H200)? 3 iterations of the 7 cm pass task, 4096 envs.
cd /home/rnallaperumal/cyclotron
source .venv/bin/activate
hostname; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
timeout 1500 ./cyclotron.sh --train --task Asimov1-Pass-R07-v0 --num_envs 4096 --headless --max_iterations 3 > outputs/test_no_rt_gpu.log 2>&1
echo "exit=$?"
grep -E 'Traceback|^\w*Error|RTX|ray tracing|not supported|Learning iteration|Iteration time|Mean episode length' outputs/test_no_rt_gpu.log | tail -15
