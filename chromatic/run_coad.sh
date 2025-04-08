#!/bin/bash
##SBATCH -p HENON
#SBATCH --nodes 1
#SBATCH --cpus-per-task 28
#SBATCH --time 48:00:00
#SBATCH --job-name coadd
#SBATCH -o ./output/logs/out.log
#SBATCH -e ./output/logs/out.log

cd $SLURM_SUBMIT_DIR
python run_coadd.py