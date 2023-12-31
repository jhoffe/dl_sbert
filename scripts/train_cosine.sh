module load python3/3.10.12
module load cuda/12.1.1

# Activate the relevant virtual environment:
source ./venv/bin/activate

export WANDB_DIR=/work3/s204071/dl_sbert/wandb/
export WANDB_CACHE_DIR=/work3/s204071/dl_sbert/wandb_cache/
export TRANSFORMERS_CACHE=/work3/s204071/dl_sbert/transformers_cache/

python main.py --batch_size=128 --precision=16-mixed --num_steps=10000 --model=distilbert-base-uncased --loss_type=cosine
python main.py --batch_size=128 --precision=16-mixed --num_steps=10000 --model=bert-base-uncased --loss_type=cosine
python main.py --batch_size=64 --precision=16-mixed --num_steps=20000 --model=bert-large-uncased --loss_type=cosine
python main.py --batch_size=32 --precision=16-mixed --num_steps=40000 --model=xlm-roberta-base --loss_type=cosine
