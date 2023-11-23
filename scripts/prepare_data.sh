#!/bin/sh
### General options
#BSUB -q hpc
#BSUB -J prepare_data
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -W 04:00
#BSUB -o logs/prepare_data_%J.out
#BSUB -e logs/prepare_data_%J.err

cd /work3/s204071/dl_sbert

module load python3/3.10.12

# Activate the relevant virtual environment:
source ./venv/bin/activate

python prepare_data.py
