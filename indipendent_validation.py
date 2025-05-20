from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset
from model import StegoNet
import torch.nn as nn
import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 64
weights_path = "best_model.pth"

val_data = np.load("stego_dataset_val.npz")
X = val_data["X"]
y = val_data["y"]

X_val_tensor = torch.tensor(X, dtype=torch.float32).permute(0, 3, 1, 2)
y_val_tensor = torch.tensor(y, dtype=torch.long)  # int labels senza unsqueeze

val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

model = StegoNet().to(device)
model.load_state_dict(torch.load(weights_path, map_location=device))
model.eval()

criterion = nn.CrossEntropyLoss()

total_loss = 0.0
correct = 0
total = 0
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)  # shape (B, 5)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * inputs.size(0)

        predicted = outputs.argmax(dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

avg_loss = total_loss / total
accuracy = correct / total
precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
conf_matrix = confusion_matrix(all_labels, all_preds)

print(f"📊 Validazione completata:")
print(f"   - Loss:     {avg_loss:.4f}")
print(f"   - Accuracy: {accuracy:.4f} ({correct}/{total})")
print(f"   - Precision:{precision:.4f}")
print(f"   - Recall:   {recall:.4f}")
print(f"   - F1-score: {f1:.4f}")
print(f"   - Confusion Matrix:\n{conf_matrix}")
