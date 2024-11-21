#!/bin/bash
#SBATCH --job-name=arts_bt # Specify job name
#SBATCH --output=arts_bt.o%j # name for standard output log file
#SBATCH --error=arts_bt.e%j # name for standard error output log
#SBATCH --partition=compute
#SBATCH --account=bm1183
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --mem=0

# Set pythonpath
export PYTHONPATH="${PYTHONPATH}:/home/m/m301049/hamp_processing/"

# execute python script in respective environment 
/home/m/m301049/.conda/envs/main/bin/python /home/m/m301049/hamp_processing/scripts/arts_calibration/arts_bt_calculation.py $1