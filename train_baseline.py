import torch
from torch.utils.data import DataLoader
from dataset_loader import SignatureDataset
from baseline_cnn import BaselineCNN
import torch.nn as nn
import torch.optim as optim

train_genuine = SignatureDataset("dataset/GPDS/train/genuine", 1)
train_forged = SignatureDataset("dataset/GPDS/train/forged", 0)

train_dataset = torch.utils.data.ConcatDataset([train_genuine, train_forged])
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

model = BaselineCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 10
for epoch in range(epochs):
    total_loss = 0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

torch.save(model.state_dict(), "models/baseline_cnn.pth")
print("Training complete. Model saved.")
