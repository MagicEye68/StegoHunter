import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F
import random
import torch

class PreprocessLSBLayer(torch.nn.Module):
    def forward(self, x):
        x_uint8 = (x * 255).to(torch.uint8)
        lsb_r = (x_uint8[:, 0, :, :] & 1).float()
        lsb_g = (x_uint8[:, 1, :, :] & 1).float()
        lsb_b = (x_uint8[:, 2, :, :] & 1).float()

        diff_r = torch.abs(lsb_r[:, :, :-1] - lsb_r[:, :, 1:])
        diff_g = torch.abs(lsb_g[:, :, :-1] - lsb_g[:, :, 1:])
        diff_b = torch.abs(lsb_b[:, :, :-1] - lsb_b[:, :, 1:])

        diff_r = F.pad(diff_r, (0, 0, 0, 1))
        diff_g = F.pad(diff_g, (0, 0, 0, 1))
        diff_b = F.pad(diff_b, (0, 0, 0, 1))

        lsb = torch.stack([diff_r, diff_g, diff_b], dim=1)
        return lsb

def extract_message_lsb_with_seed(image: np.ndarray, rgb: bool, channel: str = None, max_chars=200) -> str:
    flat_image = image.reshape(-1, 3)
    num_pixels = flat_image.shape[0]

    seed_bits = []
    for i in range(16):
        pixel_idx = i // 3
        channel_idx = i % 3
        seed_bits.append(str(flat_image[pixel_idx][channel_idx] & 1))
    seed = int(''.join(seed_bits), 2)

    if rgb:
        total_channels = num_pixels * 3
        indices = list(range(16, total_channels))
        random.seed(seed)
        random.shuffle(indices)

        bits = []
        for i in range(max_chars * 8):
            if i >= len(indices): break
            idx = indices[i]
            pixel_idx = idx // 3
            channel_idx = idx % 3
            bits.append(str(flat_image[pixel_idx][channel_idx] & 1))

    else:
        channel = channel.lower()
        channel_map = {'r': 0, 'g': 1, 'b': 2}
        ch_idx = channel_map[channel]

        indices = list(range(16, num_pixels))
        random.seed(seed)
        random.shuffle(indices)

        bits = []
        for i in range(max_chars * 8):
            if i >= len(indices): break
            pixel_idx = indices[i]
            bits.append(str(flat_image[pixel_idx][ch_idx] & 1))

    chars = []
    for i in range(0, len(bits), 8):
        byte = bits[i:i+8]
        if len(byte) < 8:
            break
        char = chr(int(''.join(byte), 2))
        chars.append(char)
        if char == '.':
            break

    return ''.join(chars)

def tensor_to_numpy_img(tensor):
    return tensor.squeeze().cpu().numpy()

def plot_images_with_lsb(original, stego, lsb_orig, lsb_stego, diff, save_path="comparison_with_lsb.png"):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    axes[0,0].imshow(original)
    axes[0,0].set_title("Original Image")
    axes[0,0].axis('off')

    axes[0,1].imshow(stego)
    axes[0,1].set_title("Stego Image")
    axes[0,1].axis('off')

    axes[0,2].imshow(diff)
    axes[0,2].set_title("Pixel Diff (Amplified)")
    axes[0,2].axis('off')

    lsb_orig_gray = np.mean(lsb_orig, axis=2)
    lsb_stego_gray = np.mean(lsb_stego, axis=2)

    axes[1,0].imshow(lsb_orig_gray, cmap='gray')
    axes[1,0].set_title("Preprocessed LSB (Original)")
    axes[1,0].axis('off')

    axes[1,1].imshow(lsb_stego_gray, cmap='gray')
    axes[1,1].set_title("Preprocessed LSB (Stego)")
    axes[1,1].axis('off')

    axes[1,2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

original_path = "img/clean_sample.png"
stego_path = "output_stego/single_vertical_b_stego.png"

original_img = np.array(Image.open(original_path).convert('RGB').resize((128,128), Image.Resampling.LANCZOS))
stego_img = np.array(Image.open(stego_path).convert('RGB'))

diff_img = np.abs(original_img.astype(np.int16) - stego_img.astype(np.int16)).astype(np.uint8)
diff_amp = np.clip(diff_img * 200, 0, 255)

preprocess = PreprocessLSBLayer()

orig_tensor = torch.tensor(original_img).permute(2,0,1).unsqueeze(0).float() / 255
stego_tensor = torch.tensor(stego_img).permute(2,0,1).unsqueeze(0).float() / 255

lsb_orig = preprocess(orig_tensor)
lsb_stego = preprocess(stego_tensor)

lsb_orig_np = np.transpose(tensor_to_numpy_img(lsb_orig), (1, 2, 0))
lsb_stego_np = np.transpose(tensor_to_numpy_img(lsb_stego), (1, 2, 0))

plot_images_with_lsb(original_img, stego_img, lsb_orig_np, lsb_stego_np, diff_amp, save_path="output_stego/diff_visualization_with_lsb.png")
print("Immagine salvata in img/diff_visualization_with_lsb.png")
