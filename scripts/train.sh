module load python3/3.10.12
module load cuda/11.8

# Activate the relevant virtual environment:
source ./venv/bin/activate

python main.py --batch_size=256 --precision=bf16-true
