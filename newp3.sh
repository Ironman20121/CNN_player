#!/bin/bash

#SBATCH -N 1                           # Request 1 node
#SBATCH -p GPU-shared                   # Specify GPU-shared partition
#SBATCH -t 48:00:00                     # Request 48 hours (adjust as needed)
#SBATCH --gpus=v100-32:1                   # Request 1 v100 GPU with 32GB memory
#SBATCH --ntasks-per-node=5             # Use all cores on the allocated node (adjust if needed)
#SBATCH --mail-type=ALL                   # Send email notifications for all job events
#SBATCH --mail-user=kundan16@hotmail.com  # Set email address for notifications
#SBATCH -o train_out3.log
#SBATCH -e train_error3.log
#SBATCH -A cis230031p


echo "Before Code activation"  >> train_out3.log
# Activate Mini Conda environment
source /ocean/projects/cis240108p/suddapal/anaconda3/etc/profile.d/conda.sh
#conda activate cllm
echo "after Code activation"  >> train_out3.log

# using conda checking if gpu takes less space

#module load anaconda
echo "Before env activation"  >> train_out3.log

conda activate project3_env
echo "After Code activation"  >> train_out3.log


# Move to your working directory
cd /ocean/projects/cis230031p/suddapal/CNN_player


echo "Before code run "  >> train_out3.log
date >> train_out3.log
srun python main.py
date >> train_out3.log
