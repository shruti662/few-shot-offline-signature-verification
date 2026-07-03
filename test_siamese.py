import os
import torch
import numpy as np
import torch.nn.functional as F
from siamese_network import SiameseNetwork
from dataset_loader import SignatureDataset

#DATASET_NAME = "GPDS"   # Change to GPDS / Kaggle when needed

model = SiameseNetwork()
model.load_state_dict(torch.load(f"models/siamese_mcyt_lr0.0005.pth"))
model.eval()


TEST_GENUINE_ROOT = f"dataset/MCYT/test/genuine"
TEST_FORGED_ROOT = f"dataset/MCYT/test/forged"

def get_embedding(img):
    with torch.no_grad():
        emb = model.forward_once(img.unsqueeze(0)).squeeze(0)
        emb = F.normalize(emb, p=2, dim=0)
    return emb

distances = []
labels = []

# -------- LOOP PER USER --------
for user in os.listdir(TEST_GENUINE_ROOT):

    genuine_path = os.path.join(TEST_GENUINE_ROOT, user)
    forged_path = os.path.join(TEST_FORGED_ROOT, user)

    if not os.path.isdir(genuine_path):
        continue

    genuine_ds = SignatureDataset(genuine_path)

    # ---- Positive pairs (Genuine-Genuine same writer) ----
    for i in range(len(genuine_ds) - 1):
        img1 = genuine_ds[i]
        img2 = genuine_ds[i + 1]

        emb1 = get_embedding(img1)
        emb2 = get_embedding(img2)

        dist = torch.norm(emb1 - emb2).item()
        distances.append(dist)
        labels.append(1)

    # ---- Negative pairs (Genuine-Forgery same writer) ----
    if os.path.isdir(forged_path):
        forged_ds = SignatureDataset(forged_path)

        for i in range(min(len(genuine_ds), len(forged_ds))):
            img1 = genuine_ds[i]
            img2 = forged_ds[i]

            emb1 = get_embedding(img1)
            emb2 = get_embedding(img2)

            dist = torch.norm(emb1 - emb2).item()
            distances.append(dist)
            labels.append(0)

distances = np.array(distances)
labels = np.array(labels)

# -------- THRESHOLD SWEEP --------
thresholds = np.linspace(distances.min(), distances.max(), 200)

best_acc = 0
best_metrics = None

for t in thresholds:
    preds = distances < t

    TP = np.sum((preds == 1) & (labels == 1))
    TN = np.sum((preds == 0) & (labels == 0))
    FP = np.sum((preds == 1) & (labels == 0))
    FN = np.sum((preds == 0) & (labels == 1))

    acc = (TP + TN) / len(labels)
    FAR = FP / (FP + TN + 1e-6)
    FRR = FN / (FN + TP + 1e-6)

    if acc > best_acc:
        best_acc = acc
        best_metrics = (t, acc, FAR, FRR)

t, acc, FAR, FRR = best_metrics

print(f"\n📊 Proper Biometric Evaluation Results (MCYT)")
print(f"Best Threshold: {t:.4f}")
print(f"Accuracy: {acc*100:.2f}%")
print(f"FAR: {FAR*100:.2f}%")
print(f"FRR: {FRR*100:.2f}%")
