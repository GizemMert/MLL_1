#!/bin/bash
#SBATCH --job-name=ge_1
#SBATCH --partition=gpu_p
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=160G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --nice=10000
#SBATCH --qos=gpu_normal
#SBATCH --constraint=a100_80gb
#SBATCH --output=my_job_output_1.log
#SBATCH --error=my_job_error_1.log

source /home/aih/gizem.mert/miniconda3/etc/profile.d/conda.sh
conda activate my_env2
# Run your job commands within the allocated resources
python train_ge_1.py