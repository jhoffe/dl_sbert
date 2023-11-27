module load python3/3.10.12
module load cuda/12.1.1

# Activate the relevant virtual environment:
source ./venv/bin/activate

python main.py --batch_size=128 --precision=16-mixed --num_steps=10000 --model=distilbert-base-uncased
python main.py --batch_size=128 --precision=16-mixed --num_steps=10000 --model=bert-base-uncased
python main.py --batch_size=128 --precision=16-mixed --num_steps=10000 --model=bert-large-uncased
python main.py --batch_size=64 --precision=16-mixed --num_steps=10000 --model=xlm-roberta-base
