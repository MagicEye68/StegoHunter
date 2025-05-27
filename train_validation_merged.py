import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
from model import StegoNet
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms

class AugmentedTensorDataset(torch.utils.data.Dataset):
    def __init__(self, data_tensor, target_tensor, transform=None):
        self.data = data_tensor
        self.targets = target_tensor
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]

        if self.transform:
            x = self.transform(x)

        return x, y


# ======= Config =======
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epochs = 50
batch_size = 64
learning_rate = 0.0005
weight_decay = 0.0001
patience = 10

# ======= Caricamento dataset =======
train_data = np.load("stego_dataset_train.npz")
X_train = torch.tensor(train_data["X"], dtype=torch.float32).permute(0, 3, 1, 2)
y_train = torch.tensor(train_data["y"], dtype=torch.long)

val_data = np.load("stego_dataset_val.npz")
X_val = torch.tensor(val_data["X"], dtype=torch.float32).permute(0, 3, 1, 2)
y_val = torch.tensor(val_data["y"], dtype=torch.long)

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomRotation(5, fill=0),
])

val_transform = None

train_dataset = AugmentedTensorDataset(X_train, y_train, transform=train_transform)
val_dataset = AugmentedTensorDataset(X_val, y_val, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

# ======= Modello =======
model = StegoNet().to(device)
if os.path.exists("best_model.pth"):
    model.load_state_dict(torch.load("best_model.pth"))
    print("Modello caricato da best_model.pth")
class_weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)


# ======= Training loop =======
best_val_loss = float("inf")
patience_counter = 0

train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

for epoch in range(num_epochs):
    # ---- Train ----
    model.train()
    train_loss, train_correct = 0.0, 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * inputs.size(0)
        train_correct += (outputs.argmax(dim=1) == labels).sum().item()

    epoch_train_loss = train_loss / len(train_loader.dataset)
    epoch_train_acc = train_correct / len(train_loader.dataset)
    train_losses.append(epoch_train_loss)
    train_accuracies.append(epoch_train_acc)

    # ---- Validation ----
    model.eval()
    val_loss, val_correct = 0.0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            val_correct += (preds == labels).sum().item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_val_loss = val_loss / len(val_loader.dataset)
    epoch_val_acc = val_correct / len(val_loader.dataset)
    scheduler.step(epoch_val_loss)
    val_losses.append(epoch_val_loss)
    val_accuracies.append(epoch_val_acc)

    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f" Train Loss: {epoch_train_loss:.4f}  Acc: {epoch_train_acc:.4f}")
    print(f" Val   Loss: {epoch_val_loss:.4f}  Acc: {epoch_val_acc:.4f}  Prec: {precision:.4f}  Recall: {recall:.4f}  F1: {f1:.4f}")

    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        patience_counter = 0
        torch.save(model.state_dict(), "best_model.pth")
        print(" --> Modello salvato (best validation loss)")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping dopo {epoch+1} epoche (val loss migliore: {best_val_loss:.4f})")
            break

# ======= Save finale =======
#torch.save(model.state_dict(), "last_model.pth")
print("Training completato. Modello migliore salvato in best_model.pth")

# ======= Plot =======
os.makedirs("img", exist_ok=True)
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.subplot(1,2,2)
plt.plot(train_accuracies, label="Train Accuracy")
plt.plot(val_accuracies, label="Validation Accuracy")
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("img/training_validation_plot.png")
plt.close()
print("Grafico salvato in img/training_validation_plot.png")
