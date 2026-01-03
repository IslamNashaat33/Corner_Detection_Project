import torch
from torch.utils.data import DataLoader
from dataset import StrokeDataset
from model import CornerBiLSTM
import numpy as np

# =====================
# LOAD TEST DATA
# =====================
TEST_DIR = "dataset_split/test"
device = "cuda" if torch.cuda.is_available() else "cpu"

test_set = StrokeDataset(TEST_DIR)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

model = CornerBiLSTM().to(device)
model.load_state_dict(torch.load("best_model.pth"))
model.eval()

tp = fp = fn = 0
total = 0

with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        preds = torch.argmax(logits, dim=-1)

        tp += ((preds == 1) & (y == 1)).sum().item()
        fp += ((preds == 1) & (y == 0)).sum().item()
        fn += ((preds == 0) & (y == 1)).sum().item()
        total += y.numel()

precision = tp / (tp + fp + 1e-8)
recall    = tp / (tp + fn + 1e-8)

print("===== TEST RESULTS =====")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
# Overall correctness and error
correct = int(total - fp - fn)
correct_pct = (correct / total * 100) if total > 0 else 0.0
error_pct = 100.0 - correct_pct
print(f"Correct:   {correct} / {int(total)} ({correct_pct:.2f}%)")
print(f"Total error: {error_pct:.2f}%")
