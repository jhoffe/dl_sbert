#!/bin/sh
### General options
### -- specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J dl_sbert
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
#BSUB -W 24:00
### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o Output_dl_sbert_train_%J.out
#BSUB -e Output_dl_sbert_train_%J.err

cd /work3/s204071/dl_sbert

module load python3/3.10.12
module load cuda/11.8

# Activate the relevant virtual environment:
source ./venv/bin/activate

python main.py --batch_size=128 --precision=16
