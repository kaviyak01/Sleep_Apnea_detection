# train_and_eval.py
import torch
from torch.utils.data import DataLoader
import numpy as np
from model import SleepApneaNet
from dataset import SleepApneaDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# Load ECG signal and labels
# ---------------------------
import wfdb

ecg_file = "D:/apnea/apnea-ecg/a01"
record = wfdb.rdrecord(ecg_file)
ann = wfdb.rdann(ecg_file, "apn")

ecg_signal = record.p_signal[:,0]  # first ECG lead
labels_bin = np.array([1 if s=="A" else 0 for s in ann.symbol])
labels_sev = np.random.randint(0, 3, size=len(labels_bin))  # dummy severity

# ---------------------------
# Dataset and DataLoader
# ---------------------------
dataset = SleepApneaDataset(ecg_signal, labels_bin, labels_sev)
loader = DataLoader(dataset, batch_size=4, shuffle=True)

# ---------------------------
# Model, Loss, Optimizer
# ---------------------------
model = SleepApneaNet().to(DEVICE)
criterion_bin = torch.nn.CrossEntropyLoss()
criterion_sev = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ---------------------------
# Training loop
# ---------------------------
EPOCHS = 3
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for x, (y_bin, y_sev) in loader:
        x, y_bin, y_sev = x.to(DEVICE), y_bin.to(DEVICE), y_sev.to(DEVICE)
        optimizer.zero_grad()
        out_bin, out_sev = model(x)
        loss = criterion_bin(out_bin, y_bin) + criterion_sev(out_sev, y_sev)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(loader):.4f}")

# ---------------------------
# Save Model
# ---------------------------
torch.save(model.state_dict(), "D:/apnea/model_demo.pt")
print("Model saved as D:/apnea/model_demo.pt")

# ---------------------------
# Evaluate Model
# ---------------------------
def evaluate_model(model, dataloader, device="cpu"):
    model.eval()
    correct_bin = 0
    correct_sev = 0
    total = 0
    with torch.no_grad():
        for x, (y_bin, y_sev) in dataloader:
            x, y_bin, y_sev = x.to(device), y_bin.to(device), y_sev.to(device)
            out_bin, out_sev = model(x)
            pred_bin = out_bin.argmax(1)
            pred_sev = out_sev.argmax(1)
            correct_bin += (pred_bin == y_bin).sum().item()
            correct_sev += (pred_sev == y_sev).sum().item()
            total += y_bin.size(0)
    print("Binary Accuracy:", correct_bin / total)
    print("Severity Accuracy:", correct_sev / total)

evaluate_model(model, loader, device=DEVICE)
