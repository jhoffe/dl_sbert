import numpy as np
import os
import torch
from sentence_transformers import SentenceTransformer, models, losses, InputExample
from dataset import MSMarcoDataset

model = SentenceTransformer("distilbert-base-nli-mean-tokens")

train_dataloader = torch.utils.data.DataLoader(
    MSMarcoDataset(), batch_size=16, shuffle=True
)

train_loss = losses.CosineSimilarityLoss(model=model)

model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, warmup_steps=100)
