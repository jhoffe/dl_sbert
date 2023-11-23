import torch
from torch import nn, Tensor
import lightning as L
from sentence_transformers import SentenceTransformer
import torchmetrics
from sentence_transformers.util import batch_to_device


class SBERT(L.LightningModule):
    def __init__(self, model: SentenceTransformer, criterion: nn.Module, lr: float = 1e-5) -> None:
        super().__init__()

        self.model = model.to(self.device)

        self.cosine = nn.CosineSimilarity()
        self.criterion = criterion
        self.lr = lr

        self.train_metrics = torchmetrics.MetricCollection({
            "train_accuracy_k1": torchmetrics.Accuracy(task="binary"),
            "train_accuracy_k3": torchmetrics.Accuracy(task="binary", top_k=3),
            "train_accuracy_k5": torchmetrics.Accuracy(task="binary", top_k=5),
        })

        self.validation_metrics = torchmetrics.MetricCollection({
            "val_accuracy_k1": torchmetrics.Accuracy(task="binary"),
            "val_accuracy_k3": torchmetrics.Accuracy(task="binary", top_k=3),
            "val_accuracy_k5": torchmetrics.Accuracy(task="binary", top_k=5),
        })

        self.test_metrics = torchmetrics.MetricCollection({
            "test_accuracy_k1": torchmetrics.Accuracy(task="binary"),
            "test_accuracy_k3": torchmetrics.Accuracy(task="binary", top_k=3),
            "test_accuracy_k5": torchmetrics.Accuracy(task="binary", top_k=5),
        })

        self.cosine_similarity_train = torchmetrics.CosineSimilarity(reduction="mean")
        self.cosine_similarity_validation = torchmetrics.CosineSimilarity(reduction="mean")
        self.cosine_similarity_test = torchmetrics.CosineSimilarity(reduction="mean")

    def forward(self, x) -> Tensor:
        tokens = self.model.tokenize(x)
        tokens = batch_to_device(tokens, self.device)
        return self.model(tokens)

    def training_step(self, batch) -> Tensor:
        x_question, x_answer, y = batch

        output_question = self(x_question)
        output_answer = self(x_answer)

        embeddings_question = output_question["sentence_embedding"]
        embeddings_answer = output_answer["sentence_embedding"]

        similarity = self.cosine(embeddings_question, embeddings_answer)
        y_hat, y = similarity.to(torch.float32), y.to(torch.float32)

        loss = self.criterion(y_hat, y)
        self.cosine_similarity_train(embeddings_question, embeddings_answer)

        self.train_metrics(y_hat, y)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_cosine_similarity", self.cosine_similarity_train, on_step=True, on_epoch=True, prog_bar=True)
        self.log_dict(self.train_metrics, on_step=True, on_epoch=True, prog_bar=False)

        return loss

    def validation_step(self, batch):
        x_question, x_answer, y = batch

        output_question = self(x_question)
        output_answer = self(x_answer)

        embeddings_question = output_question["sentence_embedding"]
        embeddings_answer = output_answer["sentence_embedding"]

        similarity = self.cosine(embeddings_question, embeddings_answer)
        y_hat, y = similarity.to(torch.float32), y.to(torch.float32)

        loss = self.criterion(y_hat, y)
        self.cosine_similarity_validation(embeddings_question, embeddings_answer)

        self.validation_metrics(y_hat, y)

        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val_cosine_similarity", self.cosine_similarity_validation, on_step=True, on_epoch=True,
                 prog_bar=False)

    def test_step(self, batch):
        x_question, x_answer, y = batch

        output_question = self(x_question)
        output_answer = self(x_answer)

        embeddings_question = output_question["sentence_embedding"]
        embeddings_answer = output_answer["sentence_embedding"]

        similarity = self.cosine(embeddings_question, embeddings_answer)
        y_hat, y = similarity.to(torch.float32), y.to(torch.float32)
        loss = self.criterion(y_hat, y)
        self.cosine_similarity_test(embeddings_question, embeddings_answer)

        self.test_metrics(y_hat, y)

        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("test_cosine_similarity", self.cosine_similarity_test, on_step=True, on_epoch=True, prog_bar=False)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
