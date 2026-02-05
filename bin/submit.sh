#! /bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --mem-per-cpu=4G
#SBATCH --time=12:00:00
#SBATCH --partition=short
#SBATCH --gres=gpu:h100:1

echo "Starting at $(date +%H:%M)"

./docker/run_singularity.sh

