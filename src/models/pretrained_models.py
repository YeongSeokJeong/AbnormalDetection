from timm.models import create_model
import torch.nn as nn
from src.models.layers import LinearBlock_with_classifier


class PretrainedModel(nn.Module):
    def __init__(self, model_name, num_layers, dropout_ratio):
        super(PretrainedModel, self).__init__()
        self.pretrained_model = create_model(model_name, pretrained=True)
        self.pretrained_model.fc = LinearBlock_with_classifier(self.pretrained_model.num_features, num_layers, dropout_ratio)
        for module in self.pretrained_model.fc.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)

    def forward(self, x):
        return self.pretrained_model(x)

