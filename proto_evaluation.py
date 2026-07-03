import os
import torch
import numpy as np
import torch.nn.functional as F
from siamese_network import SiameseNetwork
from dataset_loader import SignatureDataset

# -----------------------------
# Load trained MCYT model
# -----------------------------
model = SiameseNetwork()
model.load_state_dict(torch.load("models/siamese_mcyt_lr0.0005.pth"))
model.eval()

# -----------------------------
# Paths (MCYT Only)
# -----------------------------
TRAIN_ROOT = "dataset/MCYT/train/genuine"
TEST_GENUINE_ROOT = "dataset/MCYT/test/genuine"
TEST_FORGED_ROOT = "dataset/MCYT/test/forged"

# -----------------------------
# Embedding Function
# -----------------------------
def get_embedding(img):
    with torch.no_grad():
        emb = model.forward_once(img.unsqueeze(0)).squeeze(0)
        emb = F.normalize(emb, p=2, dim=0)  # L2 normalization
    return emb

distances = []
labels = []

# -----------------------------
# Iterate over each user
# -----------------------------
for user in os.listdir(TRAIN_ROOT):

    train_user_path = os.path.join(TRAIN_ROOT, user)
    test_genuine_path = os.path.join(TEST_GENUINE_ROOT, user)
    test_forged_path = os.path.join(TEST_FORGED_ROOT, user)

    if not os.path.isdir(train_user_path):
        continue

    # Load 5-shot training samples
    train_ds = SignatureDataset(train_user_path)

    # Compute user prototype
    embeddings = []
    for img in train_ds:
        emb = get_embedding(img)
        embeddings.append(emb)

    if len(embeddings) == 0:
        continue
    
    embeddings = torch.stack(embeddings)

    prototype = embeddings.mean(dim=0)
    prototype = F.normalize(prototype, p=2, dim=0)

    # -------------------------
    # Test Genuine Signatures
    # -------------------------
    if os.path.isdir(test_genuine_path):
        test_genuine_ds = SignatureDataset(test_genuine_path)

        for img in test_genuine_ds:
            emb = get_embedding(img)
            similarity = torch.dot(emb, prototype).item()
            distance = 1 - similarity

            distances.append(distance)
            labels.append(1)  # Genuine

    # -------------------------
    # Test Forged Signatures
    # -------------------------
    if os.path.isdir(test_forged_path):
        test_forged_ds = SignatureDataset(test_forged_path)

        for img in test_forged_ds:
            emb = get_embedding(img)
            similarity = torch.dot(emb, prototype).item()
            distance = 1 - similarity

            distances.append(distance)
            labels.append(0)  # Forged

# Convert to numpy
distances = np.array(distances)
labels = np.array(labels)

# -----------------------------
# Threshold Sweep
# -----------------------------
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

print("\n📊 Prototype Evaluation Results (MCYT)")
print(f"Best Threshold: {t:.4f}")
print(f"Accuracy: {acc*100:.2f}%")
print(f"FAR: {FAR*100:.2f}%")
print(f"FRR: {FRR*100:.2f}%")