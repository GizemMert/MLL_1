#!/bin/bash
#SBATCH --job-name=m_n
#SBATCH --partition=gpu_p
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=150G
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --nice=10000
#SBATCH --qos=gpu_normal

#SBATCH --output=my_job_output_3.log
#SBATCH --error=my_job_error_3.log

source /home/aih/gizem.mert/miniconda3/etc/profile.d/conda.sh
conda activate my_env2

python Trajectory_lung_blood.py