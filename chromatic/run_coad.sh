#!/bin/bash
##SBATCH -p HENON
#SBATCH --nodes 1
#SBATCH --cpus-per-task 28
#SBATCH --time 48:00:00
#SBATCH --job-name coadd
#SBATCH -o ./output/log_files/out.txt
#SBATCH -e ./output/log_files/out.txt

cd $SLURM_SUBMIT_DIR
python run_coadd.py