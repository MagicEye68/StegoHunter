from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc, classification_report
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from model import StegoNet
import torch.nn as nn
import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
import os
from torchsummary import summary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 64
weights_path = "best_model.pth"

val_data = np.load("stego_dataset_val.npz")
X = val_data["X"]
y = val_data["y"]

# Conta le occorrenze per label
label_counts = Counter(y)
print("Numero di elementi per label:")
for label, count in label_counts.items():
    print(f"Label {label}: {count}")

X_val_tensor = torch.tensor(X, dtype=torch.float32).permute(0, 3, 1, 2)
y_val_tensor = torch.tensor(y, dtype=torch.long)

val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

model = StegoNet().to(device)
model.load_state_dict(torch.load(weights_path, map_location=device))
model.eval()
summary(model, input_size=(3, X.shape[1], X.shape[2]))

criterion = nn.CrossEntropyLoss()

total_loss = 0.0
correct = 0
total = 0
all_preds = []
all_labels = []
all_probs = []

with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * inputs.size(0)

        probs = nn.Softmax(dim=1)(outputs)
        predicted = outputs.argmax(dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

avg_loss = total_loss / total
accuracy = correct / total
precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
conf_matrix = confusion_matrix(all_labels, all_preds)

class_names = ['clean', 'rgbH', 'rH', 'gH', 'bH', 'rgbV', 'rV', 'gV', 'bV']

os.makedirs('img/errors', exist_ok=True)

# -----------------------------
# Grafico principale (summary)
# -----------------------------
errors_per_class = conf_matrix.sum(axis=1) - np.diag(conf_matrix)

plt.figure(figsize=(16, 12))

plt.subplot(2, 2, 1)
metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
values = [accuracy, precision, recall, f1]
sns.barplot(x=metrics, y=values, hue=metrics, palette='viridis', legend=False)
plt.ylim(0, 1)
plt.title('Performance Metrics')
for i, v in enumerate(values):
    plt.text(i, v + 0.02, f"{v:.2f}", ha='center', fontweight='bold')

plt.subplot(2, 2, 2)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')

plt.subplot(2, 2, 3, polar=True)
radar_metrics = np.array(values)
angles = np.linspace(0, 2 * np.pi, len(radar_metrics), endpoint=False).tolist()
radar_metrics = np.concatenate((radar_metrics, [radar_metrics[0]]))
angles += angles[:1]
plt.polar(angles, radar_metrics, 'o-', linewidth=2)
plt.fill(angles, radar_metrics, alpha=0.25)
plt.xticks(angles[:-1], metrics)
plt.ylim(0, 1)
plt.title('Radar Plot of Metrics')

plt.subplot(2, 2, 4)
sns.barplot(x=class_names, y=errors_per_class, hue=class_names, palette='magma', legend=False)
plt.title('Errors per Class')
for i, v in enumerate(errors_per_class):
    plt.text(i, v + 5, f"{v}", ha='center', fontweight='bold')
plt.ylabel('Number of Errors')

plt.tight_layout()
plt.suptitle('Validation Report Summary', fontsize=16, fontweight='bold', y=1.02)
plt.savefig('img/validation_report.png', bbox_inches='tight', dpi=300)
plt.close()
print("✅ Report salvato come 'validation_report.png'")

# -----------------------------
# ROC Curve (binary: clean vs stego)
# -----------------------------
binary_labels = [0 if lbl == 0 else 1 for lbl in all_labels]
binary_preds = [0 if pred == 0 else 1 for pred in all_preds]
fpr, tpr, thresholds = roc_curve(binary_labels, binary_preds)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Binary ROC Curve (Clean vs. Stego)')
plt.legend(loc='lower right')
plt.savefig('img/roc_auc.png', bbox_inches='tight', dpi=300)
plt.close()
print("✅ ROC curve salvata come 'roc_auc.png'")

# -----------------------------
# Metrics per class
# -----------------------------
report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
df_report = pd.DataFrame(report).transpose()
metrics_per_class = df_report.loc[class_names, ['precision', 'recall', 'f1-score']]

metrics_per_class.plot(kind='bar', figsize=(12, 8), ylim=(0, 1), colormap='Set2')
plt.title('Metrics per Class')
plt.ylabel('Score')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('img/metrics_per_class.png', bbox_inches='tight', dpi=300)
plt.close()
print("✅ Metrics per class salvata come 'metrics_per_class.png'")

# -----------------------------
# Error analysis: immagini più sbagliate
# -----------------------------
errors = []
for idx, (true_lbl, pred_lbl, prob) in enumerate(zip(all_labels, all_preds, all_probs)):
    if true_lbl != pred_lbl:
        confidence = max(prob)
        errors.append((idx, true_lbl, pred_lbl, confidence))

# Ordina per confidenza (più alto = più sicuro ma sbagliato)
errors_sorted = sorted(errors, key=lambda x: -x[3])[:10]  # top 10

for i, (idx, true_lbl, pred_lbl, conf) in enumerate(errors_sorted):
    img = X[idx]
    plt.imshow(img.astype(np.uint8))
    plt.axis('off')
    plt.title(f"True: {class_names[true_lbl]}, Pred: {class_names[pred_lbl]}, Conf: {conf:.2f}")
    plt.savefig(f'img/errors/error_{i+1}.png', bbox_inches='tight', dpi=150)
    plt.close()

print(f"✅ Salvate {len(errors_sorted)} immagini sbagliate più confidenti in 'img/errors/'")

# -----------------------------
# Summary stampato
# -----------------------------
print(f"📊 Validazione completata:")
print(f"   - Loss:     {avg_loss:.4f}")
print(f"   - Accuracy: {accuracy:.4f} ({correct}/{total})")
print(f"   - Precision:{precision:.4f}")
print(f"   - Recall:   {recall:.4f}")
print(f"   - F1-score: {f1:.4f}")
print(f"   - Confusion Matrix:\n{conf_matrix}")
