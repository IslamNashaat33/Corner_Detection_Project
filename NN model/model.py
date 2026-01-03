import torch
import torch.nn as nn

class CornerBiLSTM(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, num_classes=2):
        super().__init__()

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )

        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        # x: (B, N, F)
        out, _ = self.lstm(x)
        logits = self.classifier(out)
        return logits
