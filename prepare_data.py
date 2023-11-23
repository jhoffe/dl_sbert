import click
import pandas as pd
import os
import webdataset as wds
from tqdm import tqdm


def load_passages(corpus_path: str) -> pd.DataFrame:
    return pd.read_csv(
        corpus_path,
        sep="\t",
        names=["passage_id", "passage"],
        dtype={"passage_id": int, "passage": str},
    ).set_index("passage_id")


def load_queries(queries_path: str) -> pd.DataFrame:
    return pd.read_csv(
        queries_path,
        sep="\t",
        names=["query_id", "query"],
        dtype={"query_id": int, "query": str},
    ).set_index("query_id")


def load_qrels(qrels_path: str) -> pd.DataFrame:
    return pd.read_csv(
        qrels_path,
        sep="\t",
        names=["query_id", "positive_passage_id", "negative_passage_id"],
        dtype={
            "query_id": int,
            "positive_passage_id": int,
            "negative_passage_id": int,
        },
    )


def load_test_qrels(qrels_path: str) -> pd.DataFrame:
    return pd.read_csv(
        qrels_path,
        sep=" ",
        names=["query_id", "Q0", "passage_id", "rating"],
        dtype={"query_id": int, "Q0": str, "passage_id": int, "rating": int},
    )


@click.command()
@click.option("--data_path", default="data/raw/", type=str)
@click.option("--output_path", default="data/processed/", type=str)
@click.option("--passages_path", default="collection.tsv", type=str)
@click.option("--queries_path", default="queries.train.tsv", type=str)
@click.option(
    "--qrels_path", default="qidpidtriples.train.full.2.tsv", type=str
)
@click.option("--subsample", default=1_000_000, type=int)
@click.option("--seed", default=42, type=int)
@click.option("--validation_fraction", default=0.1, type=float)
def prepare_data(
    data_path: str,
    output_path: str,
    passages_path: str,
    queries_path: str,
    qrels_path: str,
    subsample: int,
    seed: int,
    validation_fraction: float,
) -> None:
    os.makedirs(output_path, exist_ok=True)

    print("Loading data")
    passages = load_passages(os.path.join(data_path, passages_path))
    queries = load_queries(os.path.join(data_path, queries_path))
    qrels = load_qrels(os.path.join(data_path, qrels_path))

    qrels_values = qrels.sample(n=subsample, random_state=seed).values
    train_qrels = qrels_values[
        : int(len(qrels_values) * (1 - validation_fraction))
    ]
    validation_qrels = qrels_values[
        int(len(qrels_values) * (1 - validation_fraction)) :
    ]

    print("Writing to train shards")
    write_shards("train", output_path, passages, queries, train_qrels)

    print("Writing to validation shards")
    write_shards(
        "validation", output_path, passages, queries, validation_qrels
    )

    print("Writing to test shards")
    test_queries = load_queries(
        os.path.join(data_path, "msmarco-test2019-queries.tsv")
    )
    test_qrels = load_test_qrels(os.path.join(data_path, "2019qrels-pass.txt"))

    write_test_shards(output_path, passages, test_queries, test_qrels)


def write_test_shards(
    output_path: str,
    passages: pd.DataFrame,
    queries: pd.DataFrame,
    test_qrels: pd.DataFrame,
):
    test_sink = wds.TarWriter(os.path.join(output_path, f"test.tar"))
    for query_id, _, passage_id, rating in tqdm(
        test_qrels.values, desc="Writing to test shards"
    ):
        query = queries.loc[query_id, "query"]
        passage = passages.loc[passage_id, "passage"]

        test_sink.write(
            {
                "__key__": f"{query_id}-{passage_id}",
                "query.pyd": query,
                "passage.pyd": passage,
                "label.cls": 0 if rating < 2 else 1,
            }
        )
    test_sink.close()


def write_shards(
    name: str,
    output_path: str,
    passages: pd.DataFrame,
    queries: pd.DataFrame,
    qrels,
):
    ONE_GIGABYTE = 1024**3
    sink = wds.ShardWriter(
        os.path.join(output_path, f"{name}-%d.tar"),
        maxsize=ONE_GIGABYTE,
        maxcount=100000,
        verbose=0,
    )

    for query_id, positive_passage_id, negative_passage_id in tqdm(
        qrels, desc="Writing to shards"
    ):
        query = queries.loc[query_id, "query"]
        positive_passage = passages.loc[positive_passage_id, "passage"]
        negative_passage = passages.loc[negative_passage_id, "passage"]

        sink.write(
            {
                "__key__": f"{query_id}-{positive_passage_id}",
                "query.pyd": query,
                "passage.pyd": positive_passage,
                "label.cls": 1,
            }
        )

        sink.write(
            {
                "__key__": f"{query_id}-{negative_passage_id}",
                "query.pyd": query,
                "passage.pyd": negative_passage,
                "label.cls": 0,
            }
        )
    sink.close()


if __name__ == "__main__":
    prepare_data()
