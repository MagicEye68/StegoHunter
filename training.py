from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import torch.nn as nn
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from model import StegoNet


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_epochs = 7
batch_size = 64
learning_rate = 0.0005
weight_decay = 0.0001
patience = 10

train_data = np.load("stego_dataset_train.npz")
X = train_data["X"]
y = train_data["y"]
print("Shape di X:", X.shape)

X_train_tensor = torch.tensor(X, dtype=torch.float32).permute(0, 3, 1, 2)
y_train_tensor = torch.tensor(y, dtype=torch.long)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=6, pin_memory=True)

model = StegoNet().to(device)
if os.path.exists("best_model.pth"):
    model.load_state_dict(torch.load("best_model.pth"))
    print("Modello caricato da best_model.pth")
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

best_train_loss = float('inf')
patience_counter = 0

train_losses = []
train_accuracies = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_acc)

    if epoch_loss < best_train_loss:
            best_train_loss = epoch_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pth")
            print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f} - Accuracy: {epoch_acc:.4f} - best till now")
    else:
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f} - Accuracy: {epoch_acc:.4f}")
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping dopo {epoch+1} epoche con loss migliore: {best_train_loss:.4f}")
            break

print("Training completato.")
torch.save(model.state_dict(), 'last_model.pth')
print("Modello finale salvato in last_model.pth")

plt.figure()
plt.plot(range(1, len(train_losses)+1), train_losses, label="Train Loss")
plt.plot(range(1, len(train_accuracies)+1), train_accuracies, label="Train Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Value")
plt.title("Training Loss e Accuracy")
plt.legend()
plt.grid(True)
plt.savefig("img/training_plot.png")
plt.close()
print("Grafico training salvato in img/training_plot.png")