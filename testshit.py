import numpy as np
import pandas as pd
import os
import torch
from sentence_transformers import (
    SentenceTransformer,
    models,
    losses,
    InputExample,
    evaluation,
)
from dataset import MSMarcoDataset, MSMarcoDatasetTest

print("Is there a GPU:", torch.cuda.is_available())

model = SentenceTransformer("distilbert-base-nli-mean-tokens")

train_dataloader = torch.utils.data.DataLoader(
    MSMarcoDataset(), batch_size=16, shuffle=True
)

test_data = MSMarcoDatasetTest()

train_loss = losses.CosineSimilarityLoss(model=model)


evaluator = evaluation.EmbeddingSimilarityEvaluator.from_input_examples(
    test_data, show_progress_bar=True
)


model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=1,
    warmup_steps=100,
    output_path="models/",
    evaluator=evaluator,
    evaluation_steps=100,
    show_progress_bar=True,
)
