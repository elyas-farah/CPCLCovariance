#!/bin/bash
#SBATCH --partition=intelsr_short
#SBATCH --time=07:00:00
#SBATCH --ntasks=1
#SBATCH -N 1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=32G

#SBATCH -o slurm.%N.%j.out   # STDOUT 
#SBATCH -e slurm.%N.%j.err   # STDERR 


export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1


module load Python/3.11.3-GCCcore-12.3.0
python /home/s59efara_hpc/covariance/full_covariance_with_jax.py