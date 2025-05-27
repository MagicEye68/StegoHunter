import os
import random
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from model import StegoNet

# --- Funzioni di supporto (bin_to_int, bits_to_message, extract_message_from_image) ---

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

    return "[ERROR] Label non riconosciuto"

# --- Funzione principale di predizione ed estrazione ---

def predict_and_extract(image_path, model, device, transform):
    if not os.path.isfile(image_path):
        print(f"Immagine non trovata: {image_path}")
        return

    img_pil = Image.open(image_path).convert("RGB")
    input_tensor = transform(img_pil).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(input_tensor)
        predicted_label = outputs.argmax(dim=1).item()

    print(f"Predicted label: {predicted_label}")

    img_np = np.array(img_pil)

    message = extract_message_from_image(img_np, predicted_label)
    if predicted_label == 0:
        print("Immagine CLEAN, nessun messaggio nascosto.")
    else:
        print(f"Messaggio estratto: {message}")

def main():
    image_path = "img/dianaStego.png"
    model_weights_path = "best_model.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = StegoNet()
    model.load_state_dict(torch.load(model_weights_path, map_location=device))
    model.to(device)

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    predict_and_extract(image_path, model, device, transform)

if __name__ == "__main__":
    main()
