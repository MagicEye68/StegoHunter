import os
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from model import StegoNet
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights_path = "best_model.pth"
val_dataset_path = "stego_dataset_val.npz"
saved_images_dir = "img/val_images_by_label"

import numpy as np
import random

def bin_to_int(bits: str) -> int:
    return int(bits, 2)

def bits_to_message(bits: list) -> str:
    chars = []
    for i in range(0, len(bits), 8):
        byte = bits[i:i+8]
        if len(byte) < 8:
            break
        char = chr(int(''.join(byte), 2))
        chars.append(char)
        if char == '.':  # terminatore messaggio
            break
    return ''.join(chars)

def extract_message_from_image(image: np.ndarray, label: int) -> str:
    h, w, c = image.shape
    if c < 3:
        raise ValueError("Image must have 3 channels (RGB)")

    flat_image = image.reshape(-1, 3)
    num_pixels = flat_image.shape[0]

    if label == 0:
        # CLEAN image, niente messaggio
        return ""

    if label == 1:
        # RGB pseudocasuale: seed + messaggio distribuiti sui 3 canali
        total_channels = num_pixels * 3

        # Estrai seed dai primi 16 bit sequenziali (3 canali)
        seed_bits = []
        for i in range(16):
            pixel_idx = i // 3
            channel_idx = i % 3
            bit = flat_image[pixel_idx][channel_idx] & 1
            seed_bits.append(str(bit))
        seed = bin_to_int(''.join(seed_bits))

        # Ricostruisci sequenza shuffle
        indices = list(range(16, total_channels))
        random.seed(seed)
        random.shuffle(indices)

        # Estrai messaggio seguendo indices (LSB)
        message_bits = []
        for idx in indices:
            pixel_idx = idx // 3
            channel_idx = idx % 3
            bit = flat_image[pixel_idx][channel_idx] & 1
            message_bits.append(str(bit))

        return bits_to_message(message_bits)

    # Per i canali singoli (2 = R, 3 = G, 4 = B)
    channel_map = {2: 0, 3: 1, 4: 2}
    if label in channel_map:
        ch_idx = channel_map[label]

        # Estrai seed primi 16 bit sequenziali solo sul canale scelto
        seed_bits = []
        for i in range(16):
            bit = flat_image[i][ch_idx] & 1
            seed_bits.append(str(bit))
        seed = bin_to_int(''.join(seed_bits))

        # Ricostruisci sequenza shuffle su pixel (non canali)
        indices = list(range(16, num_pixels))
        random.seed(seed)
        random.shuffle(indices)

        # Estrai messaggio seguendo sequenza shuffle, sul canale scelto
        message_bits = []
        for pixel_idx in indices:
            bit = flat_image[pixel_idx][ch_idx] & 1
            message_bits.append(str(bit))

        return bits_to_message(message_bits)

    # Label non riconosciuto
    return "[ERROR] Label non riconosciuto"

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
    if len(saved_labels) == 9:
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
for label in range(9):
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
    elif predicted_label in [1, 2, 3, 4, 5, 6, 7, 8]:
        label_desc = {
            1: "Stego RGB (pseudocasuale): estrazione da tutti e tre i canali con seed",
            2: "Stego R (pseudocasuale): estrazione solo dal canale RED con seed",
            3: "Stego G (pseudocasuale): estrazione solo dal canale GREEN con seed",
            4: "Stego B (pseudocasuale): estrazione solo dal canale BLUE con seed",
            5: "Stego RGBV (pseudocasuale): estrazione da tutti e tre i canali con seed",
            6: "Stego RV (pseudocasuale): estrazione solo dal canale RED con seed",
            7: "Stego GV (pseudocasuale): estrazione solo dal canale GREEN con seed",
            8: "Stego BV (pseudocasuale): estrazione solo dal canale BLUE con seed",
        }
        print(label_desc[predicted_label])
        message = extract_message_from_image(img_np, predicted_label)
    else:
        print("Label non riconosciuta.")
        message = ""

    print(f"Messaggio estratto: {message}")