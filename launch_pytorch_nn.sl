#!/bin/bash

#SBATCH -A <ProjectID>
#SBATCH -J pytorch_tutorial_test
#SBATCH -o logs/pytorch_tutorial_-%j.o
#SBATCH -e logs/pytorch_tutorial_-%j.e
#SBATCH -t 00:20:00
#SBATCH -p batch
#SBATCH -N 2

# Only necessary if submitting like: sbatch --export=NONE ... (recommended)
# Do NOT include this line when submitting without --export=NONE
unset SLURM_EXPORT_ENV

# Load modules
module load PrgEnv-gnu/8.5.0
module load rocm/6.0.0
module load craype-accel-amd-gfx90a
module load miniforge3/23.11.0-0

# Activate your environment
source activate /path/to/my_env

# Get address of head node
export MASTER_ADDR=$(hostname -i)

# Needed to bypass MIOpen, Disk I/O Errors
export MIOPEN_USER_DB_PATH="/tmp/my-miopen-cache"
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}
rm -rf ${MIOPEN_USER_DB_PATH}
mkdir -p ${MIOPEN_USER_DB_PATH}

mkdir logs

# Run script
srun -N2 -n16 -c7 --gpus-per-task=1 --gpu-bind=closest python3 -W ignore -u ./multinode_olcf.py 2000 10 --master_addr=$MASTER_ADDR --master_port=3442