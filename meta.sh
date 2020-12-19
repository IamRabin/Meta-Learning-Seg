#!/bin/bash
#SBATCH --job-name=Meta
#SBATCH -p dgx2q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=01-05:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH -o output/slurm.%N.%j.out
#SBATCH -e output/slurm.%N.%j.err


echo "== Starting run at $(date)"
echo "== Job ID: ${SLURM_JOBID}"

mkdir -p ~/output
srun python main.py --alg=iMAML
