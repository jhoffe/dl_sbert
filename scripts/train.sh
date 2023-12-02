module load python3/3.10.12
module load cuda/12.1.1

# Activate the relevant virtual environment:
source ./venv/bin/activate

export WANDB_DIR=/work3/s204071/dl_sbert/wandb/
export WANDB_CACHE_DIR=/work3/s204071/dl_sbert/wandb_cache/
export TRANSFORMERS_CACHE=/work3/s204071/dl_sbert/transformers_cache/
export PATH=/sbin:$PATH

python main.py --batch_size=128 --precision=16-mixed --num_steps=10000 --model=distilbert-base-uncased
python main.py --batch_size=128 --precision=16-mixed --num_steps=10000 --model=bert-base-uncased
python main.py --batch_size=128 --precision=16-mixed --num_steps=10000 --model=bert-large-uncased
python main.py --batch_size=128 --precision=16-mixed --num_steps=10000 --model=roberta-base
python main.py --batch_size=128 --precision=16-mixed --num_steps=10000 --model=roberta-large
python main.py --batch_size=128 --precision=16-mixed --num_steps=10000 --model=xlm-roberta-base
python main.py --batch_size=128 --precision=16-mixed --num_steps=10000 --model=xlm-roberta-large
