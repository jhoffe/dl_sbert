from torch.utils.data import Dataset
import pandas as pd
import numpy as np


class MSMarcoDataset(Dataset):
    def __init__(
            self,
            qrels_path: str = "data/qrels.dev.small.tsv",
            queries_path: str = "data/queries.dev.small.tsv",
            passages_path: str = "data/collection.tsv",
            tokenizer=None,
    ):
        self.qrels_path = qrels_path
        self.queries_path = queries_path
        self.passages_path = passages_path

        self.tokenizer = tokenizer

        qrels_table = pd.read_csv(
            self.qrels_path,
            sep="\t",
            header=None,
            names=["query_id", "it", "passage_id", "label"],
        )
        self.qrels = qrels_table[["query_id", "passage_id", "label"]].values.tolist()

        queries_table = pd.read_csv(
            self.queries_path, sep="\t", header=None, names=["query_id", "query"]
        )
        self.queries = {
            query_id: query for query_id, query in queries_table.values.tolist()
        }

        passages_table = pd.read_csv(
            self.passages_path, sep="\t", header=None, names=["passage_id", "passage"]
        )
        self.passages = {
            passage_id: passage
            for passage_id, passage in passages_table.values.tolist()
        }

    def __len__(self) -> int:
        return len(self.qrels)

    def __getitem__(self, idx: int):
        query_id, passage_id, label = self.qrels[idx]

        return self.tokenizer.tokenize(self.queries[query_id]), self.tokenizer.tokenize(
            self.passages[passage_id]), float(label)


class MSMarcoDatasetTest(Dataset):
    def __init__(
            self,
            dataset_path: str = "data/msmarco-passagetest2019-top1000.tsv",
            tokenizer=None
    ):
        self.tokenizer = tokenizer
        self.ds_path = dataset_path

        self.ds_table = pd.read_csv(
            self.ds_path,
            sep="\t",
            header=None,
            names=["query_id", "passage_id", "query", "passage"],
        )

        self.passages = self.ds_table["passage"].tolist()

        self.queries = self.ds_table["query"].tolist()

        self.scores = np.ones(len(self.passages))

    def __len__(self) -> int:
        return len(self.passages)

    def __getitem__(self, idx: int) -> tuple[str, str, float]:
        return self.tokenizer.tokenize(self.queries[idx]), self.tokenizer.tokenize(
            self.passages[idx]
        ), 1.0
