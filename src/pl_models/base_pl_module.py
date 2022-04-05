import pytorch_lightning as pl
from torch.optim import Adam
import torch
from sklearn.metrics import f1_score


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

        acc = sum(torch.argmax(pred, dim=1) == label)/len(img)

        self.log('train/loss', loss, prog_bar=True, on_epoch=True)
        self.log('train/acc', acc, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        img, label = batch
        pred = self.forward(img)
        loss = self.valid_loss(pred, label)

        acc = sum(torch.argmax(pred, dim=1) == label)/len(img)

        self.log('valid/loss', loss, prog_bar=True, on_epoch=True)
        self.log('valid/acc', acc, on_epoch=True)
        return torch.argmax(pred, dim=1), label, loss

    def validation_step_end(self, output_list):
        valid_pred = output_list[0]
        valid_label = output_list[1]
        valid_f1_score = f1_score(valid_label.detach().cpu().numpy(),
                                  valid_pred.detach().cpu().numpy(),
                                  average='macro')
        self.log('valid/f1_score', valid_f1_score)


