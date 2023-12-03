#!/bin/sh
### General options
### -- specify queue --
#BSUB -q gpua100
### -- set the job Name --
#BSUB -J BERT
### -- ask for number of cores (default: 1) --
#BSUB -n 16
#BSUB -gpu "num=1:mode=exclusive_process"

### -- specify that the cores must be on the same host --
#BSUB -R "span[hosts=1]"
### -- specify that we need 4GB of memory per core/slot --
#BSUB -R "rusage[mem=4GB]"
### -- specify that we want the job to get killed if it exceeds 5 GB per core/slot --
#BSUB -M 5GB
### -- set walltime limit: hh:mm --
#BSUB -W 02:00
### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o logs/BERT_%J.out
#BSUB -e logs/BERT_%J.err

cd /work3/s204071/dl_sbert

module load python3/3.10.12
module load cuda/12.1.1

# Activate the relevant virtual environment:
source ./venv/bin/activate

export WANDB_DIR=/work3/s204071/.cache/wandb/
export WANDB_DATA_DIR=/work3/s204071/.cache/wandb_data/
export WANDB_CACHE_DIR=/work3/s204071/.cache/wandb_cache/
export TRANSFORMERS_CACHE=/work3/s204071/.cache/transformers_cache/
export PATH=/sbin:$PATH
export TORCH_HOME=/work3/s204071/.cache/torch/

python main.py --batch_size=64 --precision=16-mixed --num_steps=10000 --model=bert-base-uncased
