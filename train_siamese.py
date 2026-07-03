import torch
from torch.utils.data import DataLoader
from siamese_dataset import SiameseSignatureDataset
from siamese_network import SiameseNetwork
from contrastive_loss import ContrastiveLoss
import torch.nn.functional as F

# -------- Dataset --------

#DATASET_NAME = "GPDS"   # Change to CEDAR / GPDS / Kaggle when needed

#train_genuine_path = f"dataset/{DATASET_NAME}/train/genuine"
#train_forged_path = f"dataset/{DATASET_NAME}/test/forged"  # forged used for negative pairs

dataset = SiameseSignatureDataset(
    "dataset/MCYT/train/genuine",
    "dataset/MCYT/test/forged"
)

loader = DataLoader(dataset, batch_size=16, shuffle=True)


# -------- Model --------
model = SiameseNetwork()

# -------- Loss Function --------
# Option 1: Contrastive Loss (Recommended for Siamese)   
criterion = ContrastiveLoss()
# Option 2: Triplet Loss (Requires triplet sampling logic in dataset)
# criterion = torch.nn.TripletMarginLoss(margin=1.0, p=2) 

# -------- Optimizer --------
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
#alternative: optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# -------- Early Stopping Parameters --------
epochs = 50
max_batches = 200  # Limit batches per epoch for faster training during development
patience = 5
best_loss = float("inf")
counter = 0

print("Training started...")
    
for epoch in range(epochs):
    model.train()
    total_loss = 0

    for x1, x2, label in loader:
        optimizer.zero_grad()

        out1, out2 = model(x1, x2)
        loss = criterion(out1, out2, label)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

        # Early stopping logic
    if avg_loss < best_loss:
        best_loss = avg_loss
        counter = 0
        torch.save(model.state_dict(), "models/siamese_bestgpds.pth")
    else:
        counter += 1

    if counter >= patience:
        print("\n⏹ Early stopping triggered.")
        break

print("\n✅ Training complete. Best model saved.")