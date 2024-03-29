#!/bin/bash
#SBATCH --job-name=n_n
#SBATCH --partition=gpu_p
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=50G
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --nice=1
#SBATCH --qos=gpu_short
#SBATCH --output=my_job_output_4.log
#SBATCH --error=my_job_error_4.log

source /home/aih/gizem.mert/miniconda3/etc/profile.d/conda.sh
conda activate my_env2

python Umap_new.py