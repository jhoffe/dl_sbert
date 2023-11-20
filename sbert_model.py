import torch
from torch import nn, Tensor
import lightning as L
from sentence_transformers import SentenceTransformer, models
import torchmetrics


class SBERT(L.LightningModule):
    def __init__(self, model: SentenceTransformer, criterion: nn.Module, lr: float = 1e-5) -> None:
        super().__init__()

        self.model = model.to(self.device)

        self.cosine = nn.CosineSimilarity()
        self.criterion = criterion
        self.lr = lr

        self.cosine_similarity_train = torchmetrics.CosineSimilarity(reduction="mean")
        self.cosine_similarity_val = torchmetrics.CosineSimilarity(reduction="mean")
        self.cosine_similarity_test = torchmetrics.CosineSimilarity(reduction="mean")

    def forward(self, x: Tensor) -> Tensor:
        tokens = self.model.tokenize(x)
        return self.model(tokens)

    def training_step(self, batch) -> Tensor:
        x_question, x_answer, y = batch

        output_question = self(x_question)
        output_answer = self(x_answer)

        print(output_question)

        embeddings_question = output_question["sentence_embedding"]
        embeddings_answer = output_answer["sentence_embedding"]

        similarity = self.cosine(embeddings_question, embeddings_answer)
        loss = self.criterion(similarity.to(torch.float32), y.to(torch.float32))
        self.cosine_similarity_train(embeddings_question, embeddings_answer)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_cosine_similarity", self.cosine_similarity_train, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch):
        x_question, x_answer, y = batch

        output_question = self(x_question)
        output_answer = self(x_answer)

        embeddings_question = output_question["sentence_embedding"]
        embeddings_answer = output_answer["sentence_embedding"]

        similarity = self.cosine(embeddings_question, embeddings_answer)
        loss = self.criterion(similarity, y)
        self.cosine_similarity_val(similarity, y)

        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val_cosine_similarity", self.cosine_similarity_val, on_step=True, on_epoch=True, prog_bar=True)

    def test_step(self, batch):
        x_question, x_answer, y = batch

        output_question = self(x_question)
        output_answer = self(x_answer)

        embeddings_question = output_question["sentence_embedding"]
        embeddings_answer = output_answer["sentence_embedding"]

        similarity = self.cosine(embeddings_question, embeddings_answer)
        loss = self.criterion(similarity, y)
        self.cosine_similarity_test(similarity, y)

        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("test_cosine_similarity", self.cosine_similarity_test, on_step=True, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

