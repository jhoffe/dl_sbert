from torch.utils.data import Dataset
import pandas as pd
import numpy as np


class MSMarcoDataset(Dataset):
    def __init__(
            self,
            qrels_path: str = "data/qrels.dev.small.tsv",
            queries_path: str = "data/queries.dev.small.tsv",
            passages_path: str = "data/collection.tsv",
            sample_negatives: bool = False,
    ):
        self.qrels_path = qrels_path
        self.queries_path = queries_path
        self.passages_path = passages_path
        self.sample_negatives = sample_negatives

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
        return len(self.qrels) * 2 if self.sample_negatives else len(self.qrels)

    def _sample_negative(self, idx: int) -> tuple[int, int, float]:
        query_id, passage_id, label = self.qrels[idx // 2]

        avail_idx = [i for i in range(len(self.qrels)) if self.qrels[i][0] != query_id]

        random_passage_id = np.random.choice(avail_idx)

        return query_id, random_passage_id, -1.0

    def __getitem__(self, idx: int):
        if idx >= len(self.qrels):
            query_id, passage_id, label = self._sample_negative(idx)
        else:
            query_id, passage_id, label = self.qrels[idx]

        return self.queries[query_id], self.passages[passage_id], float(label)


class MSMarcoDatasetTest(Dataset):
    def __init__(
            self,
            dataset_path: str = "data/msmarco-passagetest2019-top1000.tsv"
    ):
        self.ds_path = dataset_path

        self.ds_table = pd.read_csv(
            self.ds_path,
            sep="\t",
            header=None,
            names=["query_id", "passage_id", "query", "passage"],
        )

        self.passages = self.ds_table["passage"].tolist()

        self.queries = self.ds_table["query"].tolist()

    def __len__(self) -> int:
        return len(self.passages)

    def __getitem__(self, idx: int) -> tuple[str, str, float]:
        return self.queries[idx], self.passages[idx], 1.0
