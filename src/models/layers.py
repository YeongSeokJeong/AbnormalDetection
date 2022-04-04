import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, hidden_feature):
        super(Classifier, self).__init__()
        self.classifier = nn.Linear(hidden_feature, 88)

    def forward(self, x):
        return self.classifier(x)


class LinearBlock(nn.Module):
    def __init__(self, in_feature, out_feature, dropout_ratio=0.2):
        super(LinearBlock, self).__init__()
        self.linear_block = nn.Sequential(
            nn.Linear(in_feature, out_feature),
            nn.BatchNorm1d(out_feature),
            nn.ReLU(),
            nn.Dropout(dropout_ratio)
        )

    def forward(self, x):
        return self.linear_block(x)


class LinearBlock_with_classifier(nn.Module):
    def __init__(self, hidden_dim, num_blocks, dropout_ratio):
        super(LinearBlock_with_classifier, self).__init__()
        linear_blocks = []
        for _ in range(num_blocks):
            linear_blocks.append(LinearBlock(hidden_dim, hidden_dim//2, dropout_ratio))
            hidden_dim = hidden_dim//2
        self.linear_blocks = nn.Sequential(*linear_blocks)
        self.classifier = Classifier(hidden_dim)

    def forward(self, x):
        x = self.linear_blocks(x)
        pred = self.classifier(x)
        return pred


