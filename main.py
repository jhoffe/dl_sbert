import os
from torch import optim, nn, utils, Tensor
import lightning as L
from sentence_transformers import SentenceTransformer, models


class SBERT(L.LightningModule):
    def __init__(self, model: models.Transformer) -> None:
        super().__init__()

        self.word_embedding_left = model
        self.word_embedding_right = model.clone()

        self.pooling_left = models.Pooling(model.get_word_embedding_dimension())
        self.pooling_right = models.Pooling(model.get_word_embedding_dimension())

    def forward(self, x: Tensor) -> Tensor:
        embedding_left = self.word_embedding_left(x)
        embedding_right = self.word_embedding_right(x)
