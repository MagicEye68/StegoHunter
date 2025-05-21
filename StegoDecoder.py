import os
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from model import StegoNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights_path = "best_model.pth"
val_dataset_path = "stego_dataset_val.npz"
saved_images_dir = "img/val_images_by_label"

# Funzioni di estrazione messaggio (uguali a prima)
def extract_message_lsb(image: np.ndarray) -> str:
    flat_image = image.reshape(-1, 3)
    bits = []
    for pixel in flat_image:
        for channel in pixel:
            bits.append(str(channel & 1))
    chars = []
    for i in range(0, len(bits), 8):
        byte = bits[i:i+8]
        char = chr(int(''.join(byte), 2))
        if char == '.':
            chars.append(char)
            break
        chars.append(char)
    return ''.join(chars)

def extract_message_from_channel(image: np.ndarray, channel_idx: int) -> str:
    channel_data = image[:, :, channel_idx]
    flat_channel = channel_data.flatten()
    bits = [str(pixel & 1) for pixel in flat_channel]
    chars = []
    for i in range(0, len(bits), 8):
        byte = bits[i:i+8]
        char = chr(int(''.join(byte), 2))
        if char == '.':
            chars.append(char)
            break
        chars.append(char)
    return ''.join(chars)

# Crea la cartella se non esiste
os.makedirs(saved_images_dir, exist_ok=True)

# Carica validation dataset
val_data = np.load(val_dataset_path)
X_val = val_data["X"]  # shape (N, H, W, 3)
y_val = val_data["y"]  # shape (N,)

# Salva un'immagine per ogni label
saved_labels = set()
for i, label in enumerate(y_val):
    if label not in saved_labels:
        img_array = (X_val[i] * 255).astype(np.uint8)
        img = Image.fromarray(img_array)
        save_path = os.path.join(saved_images_dir, f"label_{label}.png")
        img.save(save_path)
        saved_labels.add(label)
    if len(saved_labels) == 5:
        break

# Setup modello e trasformazione input
model = StegoNet()
model.load_state_dict(torch.load(weights_path, map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
])

# Per ogni immagine salvata: carica, predici, estrai messaggio
for label in range(5):
    img_path = os.path.join(saved_images_dir, f"label_{label}.png")
    if not os.path.isfile(img_path):
        print(f"Immagine per label {label} non trovata, salto.")
        continue

    img_pil = Image.open(img_path).convert("RGB")
    input_tensor = transform(img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        predicted_label = outputs.argmax(dim=1).item()

    print(f"\nImmagine salvata per label {label}")
    print(f"Predicted label: {predicted_label}")

    img_np = np.array(img_pil)

    if predicted_label == 0:
        print("Immagine CLEAN, nessun messaggio nascosto.")
        message = ""
    elif predicted_label == 1:
        print("Stego RGB: estrazione messaggio da tutti e tre i canali")
        message = extract_message_lsb(img_np)
    elif predicted_label == 2:
        print("Stego R: estrazione messaggio dal canale RED")
        message = extract_message_from_channel(img_np, 0)
    elif predicted_label == 3:
        print("Stego G: estrazione messaggio dal canale GREEN")
        message = extract_message_from_channel(img_np, 1)
    elif predicted_label == 4:
        print("Stego B: estrazione messaggio dal canale BLUE")
        message = extract_message_from_channel(img_np, 2)
    else:
        message = "Label non riconosciuta."

    print(f"Messaggio estratto: {message}")
