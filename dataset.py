from torch.utils.data import Dataset
import pandas as pd


class MSMarcoDataset(Dataset):
    def __init__(self, qrels_path: str = "data/qrels.dev.small.tsv", queries_path: str = "data/queries.dev.small.tsv",
                 passages_path: str = "data/collection.tsv"):
        self.qrels_path = qrels_path
        self.queries_path = queries_path
        self.passages_path = passages_path

        qrels_table = pd.read_csv(self.qrels_path, sep="\t", header=None,
                                  names=["query_id", "it", "passage_id", "label"])
        self.qrels = qrels_table[["query_id", "passage_id", "label"]].values.tolist()

        queries_table = pd.read_csv(self.queries_path, sep="\t", header=None, names=["query_id", "query"])
        self.queries = {query_id: query for query_id, query in queries_table.values.tolist()}

        passages_table = pd.read_csv(self.passages_path, sep="\t", header=None, names=["passage_id", "passage"])
        self.passages = {passage_id: passage for passage_id, passage in passages_table.values.tolist()}

    def __len__(self) -> int:
        return len(self.qrels)

    def __getitem__(self, idx: int) -> tuple[str, str, int]:
        query_id, passage_id, label = self.qrels[idx]

        return self.queries[query_id], self.passages[passage_id], label
