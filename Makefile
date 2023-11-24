# Commands for dispatching

dev:
	python main.py --batch_size=4 --dev

submit-a100:
	bsub < scripts/train_a100.sh

submit-v100:
	bsub < scripts/train_v100.sh