#!/bin/bash
#SBATCH --job-name=ge_1
#SBATCH --partition=gpu_p
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=200G
#SBATCH --gres=gpu:1
#SBATCH --time=5:00:00
#SBATCH --nice=10000
#SBATCH --qos=gpu_long

#SBATCH --output=my_job_output_1.log
#SBATCH --error=my_job_error_1.log

source /home/aih/gizem.mert/miniconda3/etc/profile.d/conda.sh
conda activate my_env2

python train4_cp4.py