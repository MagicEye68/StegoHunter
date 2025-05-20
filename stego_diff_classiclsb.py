import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

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

def plot_images(original, stego, diff, save_path="comparison.png"):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(original)
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Stego Image")
    plt.imshow(stego)
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Pixel Diff")
    plt.imshow(diff)
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# ---- MAIN SCRIPT ----
original_path = "img/clean_sample.png"
stego_path = "img/stego_sample.png"

original_img = np.array(Image.open(original_path).convert('RGB'))
stego_img = np.array(Image.open(stego_path).convert('RGB'))

diff_img = np.abs(original_img.astype(np.int16) - stego_img.astype(np.int16)).astype(np.uint8)
diff_amp = diff_img * 20

message = extract_message_lsb(np.array(Image.open("img/stego_sample.png")))
print(f"🕵️ Messaggio estratto: '{message}'")

plot_images(original_img, stego_img, diff_amp, save_path="img/diff_visualization_classiclsb.png")
print("🖼️ Immagine salvata in img/diff_visualization_classiclsb.png")
