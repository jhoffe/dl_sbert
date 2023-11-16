#!/bin/bash

module load python3/3.10.12
module load cuda/11.8

# Create the env
python -m venv venv

source venv/bin/activate

# Install the requirements
pip install -r requirements.txt
