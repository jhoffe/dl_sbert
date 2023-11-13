import torch
from sentence_transformers import (
    SentenceTransformer,
    losses,
    evaluation,
)
from dataset import MSMarcoDataset, MSMarcoDatasetTest
from torch.utils.data import Subset

print("Is there a GPU:", torch.cuda.is_available())

model = SentenceTransformer("distilbert-base-nli-mean-tokens")

train_dataloader = torch.utils.data.DataLoader(
    MSMarcoDataset(), batch_size=16, shuffle=True
)
test_data = MSMarcoDatasetTest()
train_loss = losses.CosineSimilarityLoss(model=model)

qid_to_query = dict(test_data.ds_table[["query_id", "query"]].values.tolist())
cid_to_doc = dict(test_data.ds_table[["passage_id", "passage"]].values.tolist())

relevant_docs_dict = (
    test_data.ds_table.groupby("query_id")
    .agg({"passage_id": set})
    .to_dict()["passage_id"]
)

relevant_docs_dict = {
    qid: set(list(pids)[:10]) for qid, pids in relevant_docs_dict.items()
}

all_doc_ids = set([d for ds in relevant_docs_dict.values() for d in ds])

# filter out docs to the relevant ones
cid_to_doc = {cid: doc for cid, doc in cid_to_doc.items() if cid in all_doc_ids}

assert len(relevant_docs_dict) == len(qid_to_query)

ir = evaluation.InformationRetrievalEvaluator(
    queries=qid_to_query,
    corpus=cid_to_doc,
    relevant_docs=relevant_docs_dict,
    batch_size=16,
    name="test",
    show_progress_bar=True,
)

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=1,
    warmup_steps=100,
    output_path="models/",
    evaluator=ir,
    evaluation_steps=100,
    show_progress_bar=True,
)
