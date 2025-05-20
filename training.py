from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import torch.nn as nn
import numpy as np
import torch


import matplotlib.pyplot as plt
from model import StegoNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epochs = 10
batch_size = 64
learning_rate = 0.001

train_data = np.load("stego_dataset_train.npz")
X = train_data["X"]
y = train_data["y"]

X_train_tensor = torch.tensor(X, dtype=torch.float32).permute(0, 3, 1, 2)
y_train_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

model = StegoNet().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

best_acc = 0.0
accuracies = []

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
        predicted = (outputs > 0.5).float()
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    accuracies.append(epoch_acc)

    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f} - Accuracy: {epoch_acc:.4f}")

    if epoch_acc > best_acc:
        best_acc = epoch_acc
        torch.save(model.state_dict(), 'best_model.pth')
        print(f"Nuovo best model salvato con accuracy: {best_acc:.4f}")

print("Training completato.")
torch.save(model.state_dict(), 'last_model.pth')
print("Modello finale salvato in last_model.pth")

plt.figure()
plt.plot(range(1, num_epochs+1), accuracies, marker='o')
plt.title("Accuratezza per epoca")
plt.xlabel("Epoca")
plt.ylabel("Accuracy")
plt.grid(True)
plt.savefig("img/accuracy_plot.png")
plt.close()
print("Grafico accuratezza salvato in img/accuracy_plot.png")