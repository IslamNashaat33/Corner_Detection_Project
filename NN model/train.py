import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from dataset import StrokeDataset
from model import CornerBiLSTM
from utils import EarlyStopping

# =====================
# CONFIG
# =====================
TRAIN_DIR = "dataset_split_v2/train"
VAL_DIR   = "dataset_split_v2/val"

EPOCHS = 10
LR = 1e-3
BATCH_SIZE = 1
PATIENCE = 7

device = "cuda" if torch.cuda.is_available() else "cpu"

# =====================
# LOAD DATA
# =====================
train_set = StrokeDataset(TRAIN_DIR)
val_set   = StrokeDataset(VAL_DIR)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

model = CornerBiLSTM().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

early_stopper = EarlyStopping(patience=PATIENCE)

# =====================
# TRAINING
# =====================
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x)

        loss = criterion(logits.view(-1, 2), y.view(-1))
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # =====================
    # VALIDATION
    # =====================
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits.view(-1, 2), y.view(-1))
            val_loss += loss.item()

    print(
        f"Epoch {epoch+1:03d} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f}"
    )

    improved = early_stopper.step(val_loss)
    if improved:
        torch.save(model.state_dict(), "best_model.pth")

    if early_stopper.stop:
        print("Early stopping triggered.")
        break

print("Training finished.")
