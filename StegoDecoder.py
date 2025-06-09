import os
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from model import StegoNet
import random
import numpy as np
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights_path = "best_model.pth"
val_dataset_path = "stego_dataset_val.npz"
saved_images_dir = "img/val_images_by_label"

def bin_to_int(bits: str) -> int:
    return int(bits, 2)

def bits_to_message(bits: list) -> str:
    chars = []
    for i in range(0, len(bits), 8):
        byte = bits[i:i+8]
        if len(byte) < 8:
            break
        char_code = int(''.join(byte), 2)
        if char_code < 32 or char_code > 126:
            break
        char = chr(char_code)
        chars.append(char)
        if char == '.':
            break
    return ''.join(chars)

def extract_message_from_image(image: np.ndarray, label: int) -> str:
    h, w, c = image.shape
    if c < 3:
        raise ValueError("Image must have 3 channels (RGB)")

    flat_image = image.reshape(-1, 3)
    num_pixels = flat_image.shape[0]

    if label == 0:
        return ""

    if label == 1:
        total_channels = num_pixels * 3

        seed_bits = []
        for i in range(16):
            pixel_idx = i // 3
            channel_idx = i % 3
            bit = flat_image[pixel_idx][channel_idx] & 1
            seed_bits.append(str(bit))
        seed = bin_to_int(''.join(seed_bits))

        indices = list(range(16, total_channels))
        random.seed(seed)
        random.shuffle(indices)

        message_bits = []
        for idx in indices:
            pixel_idx = idx // 3
            channel_idx = idx % 3
            bit = flat_image[pixel_idx][channel_idx] & 1
            message_bits.append(str(bit))

        return bits_to_message(message_bits)

    channel_map = {2: 0, 3: 1, 4: 2}
    if label in channel_map:
        ch_idx = channel_map[label]

        seed_bits = []
        for i in range(16):
            bit = flat_image[i][ch_idx] & 1
            seed_bits.append(str(bit))
        seed = bin_to_int(''.join(seed_bits))

        indices = list(range(16, num_pixels))
        random.seed(seed)
        random.shuffle(indices)

        message_bits = []
        for pixel_idx in indices:
            bit = flat_image[pixel_idx][ch_idx] & 1
            message_bits.append(str(bit))

        return bits_to_message(message_bits)

    if label in [5, 6, 7, 8]:
        h, w, c = image.shape
        if c < 3:
            raise ValueError("Image must have 3 channels (RGB)")

        bits = []

        if label == 5:
            section_w = w // 3
            remainder = w % 3
            total_bits = h * w * 3
            max_len = h * section_w

            def extract_vertical_bits_limited(section_start, section_end, channel_idx, max_bits):
                bits = []
                for col in range(section_start, section_end):
                    for row in range(h):
                        if len(bits) >= max_bits:
                            return bits
                        pixel = image[row, col]
                        bits.append(str(pixel[channel_idx] & 1))
                return bits

            part_len = max_len
            part_r = extract_vertical_bits_limited(0, section_w, 0, part_len)
            part_g = extract_vertical_bits_limited(section_w, 2 * section_w, 1, part_len)
            part_b = extract_vertical_bits_limited(2 * section_w, w, 2, len(image) * 3 - 2 * part_len)

            message_bits = part_r + part_g + part_b

            return bits_to_message(message_bits)


        else:
            channel_map = {6: ('r', 0), 7: ('g', 1), 8: ('b', 2)}
            channel_name, ch_idx = channel_map[label]

            if channel_name == 'r':
                start_col = 0
            elif channel_name == 'g':
                start_col = w // 3
            else:
                start_col = (2 * w) // 3

            for col in range(start_col, w):
                for row in range(h):
                    pixel = image[row, col]
                    bits.append(str(pixel[ch_idx] & 1))

        return bits_to_message(bits)

    return "[ERROR] Label non riconosciuto"

os.makedirs(saved_images_dir, exist_ok=True)

val_data = np.load(val_dataset_path)
X_val = val_data["X"]
y_val = val_data["y"]

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

model = StegoNet()
model.load_state_dict(torch.load(weights_path, map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
])

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