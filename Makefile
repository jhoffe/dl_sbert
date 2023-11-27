download:
	wget https://msmarco.z22.web.core.windows.net/msmarcoranking/qidpidtriples.train.full.2.tsv.gz data/raw/qidpidtriples.train.full.2.tsv.gz
	wget https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-test2019-queries.tsv.gz data/raw/msmarco-test2019-queries.tsv.gz
	wget https://msmarco.z22.web.core.windows.net/msmarcoranking/collection.tar.gz data/raw/collection.tar.gz
	wget https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-passagetest2019-top1000.tsv.gz data/raw/msmarco-passagetest2019-top1000.tsv.gz
	wget https://trec.nist.gov/data/deep/2019qrels-pass.txt data/raw/2019qrels-pass.txt
	wget https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-test2019-queries.tsv.gz data/raw/msmarco-test2019-queries.tsv.gz
	gzip -d data/raw/qidpidtriples.train.full.2.tsv.gz
	gzip -d data/raw/msmarco-test2019-queries.tsv.gz
	gzip -d data/raw/msmarco-passagetest2019-top1000.tsv.gz
	tar -xvf data/raw/collection.tar.gz -C data/raw/


prepare:
	python prepare_data.py

dev:
	python main.py --batch_size=4 --dev

submit-a100:
	bsub < scripts/train_a100.sh

submit-v100:
	bsub < scripts/train_v100.sh