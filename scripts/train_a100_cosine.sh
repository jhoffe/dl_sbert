#!/bin/sh
### General options
### -- specify queue --
#BSUB -q gpua100
### -- set the job Name --
#BSUB -J dl_sbert_embedding_loss
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
#BSUB -o logs/Output_dl_sbert_train_cosine_%J.out
#BSUB -e logs/Output_dl_sbert_train_cosine_%J.err
#BSUB -R "select[gpu80gb]"

cd /work3/s204071/dl_sbert
./scripts/train_cosine.sh