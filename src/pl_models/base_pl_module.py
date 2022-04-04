import pytorch_lightning as pl
from torch.optim import Adam
import torch


class BasePlModule(pl.LightningModule):
    def __init__(self, model, lr, train_loss, valid_loss):
        super(BasePlModule, self).__init__()
        self.model = model
        self.lr = lr

        self.train_loss = train_loss
        self.valid_loss = valid_loss

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr = self.lr)

    def training_step(self, batch, batch_idx):
        img, label = batch
        pred = self.forward(img)
        loss = self.train_loss(pred, label)

        acc = sum(torch.argmax(pred, dim=1) == label)

        self.log('train/loss', loss)
        self.log('train/acc', acc)
        return loss

    def validation_step(self, batch, batch_idx):
        img, label = batch
        pred = self.forward(img)
        loss = self.valid_loss(pred, label)

        acc = sum(torch.argmax(pred, dim=1) == label)

        self.log('valid/loss', loss)
        self.log('valid/acc', acc)
        return loss