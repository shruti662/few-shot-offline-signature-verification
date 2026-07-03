import torch
from torch.utils.data import DataLoader
from dataset_loader import SignatureDataset
from baseline_cnn import BaselineCNN
from sklearn.metrics import confusion_matrix

test_genuine = SignatureDataset("dataset/GPDS/test/genuine", 1)
test_forged = SignatureDataset("dataset/GPDS/test/forged", 0)

test_dataset = torch.utils.data.ConcatDataset([test_genuine, test_forged])
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

model = BaselineCNN()
model.load_state_dict(torch.load("models/baseline_cnn.pth"))
model.eval()

y_true, y_pred = [], []

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)
        y_true.extend(labels.tolist())
        y_pred.extend(preds.tolist())

tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

accuracy = (tp + tn) / (tp + tn + fp + fn)
far = fp / (fp + tn)
frr = fn / (fn + tp)

print(f"Accuracy: {accuracy*100:.2f}%")
print(f"FAR: {far*100:.2f}%")
print(f"FRR: {frr*100:.2f}%")
