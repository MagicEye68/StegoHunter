from torch.utils.data import Dataset
from torchvision import datasets, transforms
from sklearn.utils import shuffle
from faker import Faker
from PIL import Image
from tqdm import tqdm
import numpy as np
import random
import torch
import cv2
import re
import os

class BossbaseDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.image_files = sorted([f for f in os.listdir(root_dir) if f.endswith(".pgm")])
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        try:
            with Image.open(img_path) as img:
                image = img.convert("RGB")
        except Exception as e:
            print(f"[SKIP] Errore aprendo {img_path}: {e}")
            return None

        if self.transform:
            image = self.transform(image)
        return image

def load_bossbase_numpy(root_dir, image_size=128, max_images=None):
    files = sorted([f for f in os.listdir(root_dir) if f.endswith(".pgm")])
    if max_images:
        files = files[:max_images]

    images = []
    for fname in tqdm(files, desc="Loading BOSSBase images"):
        path = os.path.join(root_dir, fname)
        try:
            img = Image.open(path).convert("RGB")
            img = img.resize((image_size, image_size), Image.LANCZOS)
            images.append(np.array(img))
        except Exception as e:
            print(f"[SKIP] Error opening {path}: {e}")

    images_np = np.stack(images).astype(np.uint8)

    images_tensor = torch.tensor(images_np / 255., dtype=torch.float32).permute(0, 3, 1, 2)

    return images_tensor

def load_cifar10_numpy(train=True, download=True, root='./data') -> 'np.ndarray':
    transform = transforms.ToTensor()
    dataset = datasets.CIFAR10(root=root, train=train, download=download, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset))
    images, _ = next(iter(loader))
    images_np = (images.numpy() * 255).astype('uint8').transpose(0, 2, 3, 1)
    return images_np

def dataset_to_numpy(dataset):
    imgs = []
    for img, _ in dataset:
        img_np = img.permute(1, 2, 0).numpy()
        imgs.append(img_np)
    return np.array(imgs)

def generate_random_sentence(max_length=3072):
    while True:
        nb_words = random.randint(40, 60)
        sentence = fake.sentence(nb_words)
        cleaned = clean_message(sentence)
        encoded = classicLSB(cleaned)
        if len(encoded) <= max_length:
            return cleaned

def clean_message(msg):
    msg = msg.lower()
    allowed = re.compile(r"[a-z ,.?!)]+")
    cleaned = ''.join(ch for ch in msg if allowed.match(ch))
    return cleaned

def int_to_bin(value: int, bit_length: int) -> str:
    return format(value, f'0{bit_length}b')

def classicLSB(message:str) -> str:
    return ''.join(format(ord(char), '08b') for char in message)

def myLSB(message:str):
    punctuation_map = {
        ',': 27,
        '.': 28,
        '?': 29,
        '!': 30,
    }
    parts = []
    for char in message:
        if 'a' <= char <= 'z':
            char_value = ord(char) - ord('a') + 1
            ones_part = '1' * char_value
            parts.append(ones_part)
        elif char == ' ':
            parts.append('0')
        elif char in punctuation_map:
            char_value = punctuation_map[char]
            ones_part = '1' * char_value
            parts.append(ones_part)
        else:
            raise ValueError(f"Carattere non valido: '{char}'. Accettate solo lettere minuscole a-z e spazi.")
    return '0'.join(parts)+'000'

def hide_message_lsb(image: np.ndarray, message: str, rgb: bool, channel: str = None, seed: int = None) -> np.ndarray:
    image = image.copy()
    h, w, c = image.shape
    if c < 3:
        raise ValueError("Image must have 3 channels (RGB)")

    flat_image = image.reshape(-1, 3)
    num_pixels = flat_image.shape[0]

    if seed is None:
        seed = random.randint(0, 65535)

    seed_bin = int_to_bin(seed, 16)
    message_bin = classicLSB(message)
    full_bin = seed_bin + message_bin
    message_len = len(full_bin)

    if rgb:
        total_channels = num_pixels * 3
        if message_len > total_channels:
            raise ValueError("Message + seed too long for image (RGB mode)")

        indices = list(range(16, total_channels))
        random.seed(seed)
        random.shuffle(indices)

        for i in range(16):
            pixel_idx = i // 3
            channel_idx = i % 3
            flat_image[pixel_idx][channel_idx] &= 0xFE
            flat_image[pixel_idx][channel_idx] |= int(seed_bin[i])

        for i in range(len(message_bin)):
            idx = indices[i]
            pixel_idx = idx // 3
            channel_idx = idx % 3
            flat_image[pixel_idx][channel_idx] &= 0xFE
            flat_image[pixel_idx][channel_idx] |= int(message_bin[i])

    else:
        channel = channel.lower()
        channel_map = {'r': 0, 'g': 1, 'b': 2}
        if channel not in channel_map:
            raise ValueError("Invalid channel. Use 'r', 'g', or 'b'.")
        ch_idx = channel_map[channel]

        if message_len > num_pixels:
            raise ValueError("Message + seed too long for selected channel")

        indices = list(range(16, num_pixels))
        random.seed(seed)
        random.shuffle(indices)

        for i in range(16):
            flat_image[i][ch_idx] &= 0xFE
            flat_image[i][ch_idx] |= int(seed_bin[i])

        for i in range(len(message_bin)):
            pixel_idx = indices[i]
            flat_image[pixel_idx][ch_idx] &= 0xFE
            flat_image[pixel_idx][ch_idx] |= int(message_bin[i])

    stego_image = flat_image.reshape((h, w, 3)).astype(np.uint8)
    return stego_image


def hide_message_lsb_sequential(image: np.ndarray, message: str, rgb: bool, channel: str = None) -> np.ndarray:
    image = image.copy()
    h, w, c = image.shape
    if c < 3:
        raise ValueError("L'immagine deve avere 3 canali (RGB)")

    message_bin = ''.join(format(ord(char), '08b') for char in message)
    message_len = len(message_bin)

    if rgb:
        total_capacity = h * w * 3
        if message_len > total_capacity:
            raise ValueError("Messaggio troppo lungo per l'immagine (modalità RGB)")

        part_len = (message_len + 2) // 3
        part_r = message_bin[:part_len]
        part_g = message_bin[part_len:2*part_len]
        part_b = message_bin[2*part_len:]

        def embed_bits_in_rows(start_row, end_row, channel_idx, bits):
            idx = 0
            for row in range(start_row, end_row):
                for col in range(w):
                    if idx >= len(bits):
                        return
                    pixel = image[row, col]
                    pixel[channel_idx] = (pixel[channel_idx] & 0xFE) | int(bits[idx])
                    image[row, col] = pixel
                    idx += 1

        section_height = h // 3
        embed_bits_in_rows(0, section_height, 0, part_r)
        embed_bits_in_rows(section_height, 2*section_height, 1, part_g)
        embed_bits_in_rows(2*section_height, h, 2, part_b)

    else:
        channel = channel.lower()
        channel_map = {'r': 0, 'g': 1, 'b': 2}
        if channel not in channel_map:
            raise ValueError("Canale non valido. Usa 'r', 'g' o 'b'.")
        ch_idx = channel_map[channel]

        start_row = {'r': 0, 'g': h // 3, 'b': 2 * h // 3}[channel]

        available_pixels = (h - start_row) * w
        if message_len > available_pixels:
            raise ValueError("Messaggio troppo lungo per il canale selezionato")

        idx = 0
        for row in range(start_row, h):
            for col in range(w):
                if idx >= message_len:
                    return image
                pixel = image[row, col]
                pixel[ch_idx] = (pixel[ch_idx] & 0xFE) | int(message_bin[idx])
                image[row, col] = pixel
                idx += 1

    return image

def hide_message_lsb_sequential_vertical(image: np.ndarray, message: str, rgb: bool, channel: str = None) -> np.ndarray:
    image = image.copy()
    h, w, c = image.shape
    if c < 3:
        raise ValueError("L'immagine deve avere 3 canali (RGB)")

    flat_image = image.reshape(-1, 3)
    num_pixels = flat_image.shape[0]
    message_bin = ''.join(format(ord(char), '08b') for char in message)
    message_len = len(message_bin)

    if rgb:
        total_capacity = h * w * 3
        if message_len > total_capacity:
            raise ValueError("Messaggio troppo lungo per l'immagine (modalità RGB verticale)")

        part_len = message_len // 3 + (1 if message_len % 3 > 0 else 0)
        part_r = message_bin[:part_len]
        part_g = message_bin[part_len:2 * part_len]
        part_b = message_bin[2 * part_len:]

        section_w = w // 3

        def embed_bits_vertical(section_start, section_end, channel_idx, bits):
            idx = 0
            for col in range(section_start, section_end):
                for row in range(h):
                    if idx >= len(bits):
                        return
                    pixel = image[row, col]
                    pixel[channel_idx] = (pixel[channel_idx] & 0xFE) | int(bits[idx])
                    image[row, col] = pixel
                    idx += 1

        embed_bits_vertical(0, section_w, 0, part_r)
        embed_bits_vertical(section_w, 2 * section_w, 1, part_g)
        embed_bits_vertical(2 * section_w, w, 2, part_b)
    else:
        channel = channel.lower()
        channel_map = {'r': 0, 'g': 1, 'b': 2}
        if channel not in channel_map:
            raise ValueError("Canale non valido. Usa 'r', 'g' o 'b'.")
        ch_idx = channel_map[channel]

        if channel == 'r':
            start_col = 0
        elif channel == 'g':
            start_col = w // 3
        else:
            start_col = (2 * w) // 3

        available_pixels = (w - start_col) * h
        if message_len > available_pixels:
            raise ValueError("Messaggio troppo lungo per il canale selezionato (verticale)")

        idx = 0
        for col in range(start_col, w):
            for row in range(h):
                if idx >= message_len:
                    break
                pixel = image[row, col]
                pixel[ch_idx] = (pixel[ch_idx] & 0xFE) | int(message_bin[idx])
                image[row, col] = pixel
                idx += 1

    return image

def generate_stego_dataset(images):
    if isinstance(images, torch.Tensor):
        images = images.permute(0, 2, 3, 1).numpy()
    if images.ndim == 4 and images.shape[1] == 3 and images.shape[-1] != 3:
        images = images.transpose(0, 2, 3, 1)
    images = np.clip(images * 255, 0, 255).astype(np.uint8)

    processed_images = []
    labels = []
    stessaFoto = []
    stessaFotoLabels = []

    stego_methods = [
        ('rgb', None),
        ('single', 'r'),
        ('single', 'g'),
        ('single', 'b'),
        ('rgb_vertical', None),
        ('single_vertical', 'r'),
        ('single_vertical', 'g'),
        ('single_vertical', 'b'),
    ]

    for i, img in enumerate(tqdm(images, desc="Generazione dataset")):
        new_size = random.randint(128, 256)
        img_resized = cv2.resize(img, (new_size, new_size), interpolation=cv2.INTER_LINEAR)
        if i == 99:
            cv2.imwrite("img/clean_sample.png", cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR))
            stessaFoto.append(img_resized)
            stessaFotoLabels.append(0)

            MESSAGE = generate_random_sentence()
            img_stego = hide_message_lsb_sequential(img_resized, message=MESSAGE, rgb=True)

            cv2.imwrite("img/stego_sample.png", cv2.cvtColor(img_stego, cv2.COLOR_RGB2BGR))
            stessaFoto.append(img_stego)
            stessaFotoLabels.append(1)

            XFotoUnica = np.array(stessaFoto, dtype=np.float32) / 255.0
            YFotoUnica = np.array(stessaFotoLabels)
            np.savez_compressed("stessaFoto.npz", X=XFotoUnica, y=YFotoUnica)

        label_type = i % 9

        if label_type == 0:
            labels.append(0)
            processed_images.append(img)
        else:
            method, channel = stego_methods[label_type - 1]

            MESSAGE = generate_random_sentence()
            if method == 'rgb':
                img_stego = hide_message_lsb(img, message=MESSAGE, rgb=True)
            elif method == 'single':
                img_stego = hide_message_lsb(img, message=MESSAGE, rgb=False, channel=channel)
            elif method == 'rgb_vertical':
             img_stego = hide_message_lsb_sequential_vertical(img, message=MESSAGE, rgb=True)
            elif method == 'single_vertical':
                img_stego = hide_message_lsb_sequential_vertical(img, message=MESSAGE, rgb=False, channel=channel)
            else:
                raise ValueError(f"Metodo stego sconosciuto: {method}")

            labels.append(label_type)
            processed_images.append(img_stego)

    X = np.array(processed_images, dtype=np.float32) / 255.0
    y = np.array(labels)

    if len(processed_images) != len(labels):
        print(f"[ERRORE] Numero immagini ({len(processed_images)}) diverso da numero etichette ({len(labels)})")

    return X, y

transform_stl10 = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor()
])

#http://dde.binghamton.edu/download/ImageDB/BOSSbase_1.01.zip
images_boss = load_bossbase_numpy("./BOSSbase_1.01", image_size=96)
#images_np = load_cifar10_numpy()
stl10_train = datasets.STL10(root="./data", split="train", download=True, transform=transform_stl10)
stl10_test = datasets.STL10(root="./data", split="test", download=True, transform=transform_stl10)
images_stl_train = dataset_to_numpy(stl10_train)
images_stl_test = dataset_to_numpy(stl10_test)
images_stl = np.concatenate((images_stl_train, images_stl_test), axis=0).transpose(0, 3, 1, 2)
num_to_keep = len(images_stl)
indices = np.random.choice(len(images_stl), num_to_keep, replace=False)
images_stl_subset = images_stl[indices]
images_np = np.concatenate((images_boss.numpy(), images_stl), axis=0)
images_np = np.concatenate((images_boss, images_stl_subset), axis=0)
images_np=shuffle(images_np, random_state=42)
print(f"Dimensione dataset finale: {images_np.shape}")
fake = Faker()
X, y = generate_stego_dataset(images_np)
X, y = shuffle(X, y, random_state=42)
split_index = int(0.8 * len(X))
X_train = X[:split_index]
y_train = y[:split_index]
X_val = X[split_index:]
y_val = y[split_index:]

np.savez_compressed("stego_dataset_train.npz", X=X_train, y=y_train)
np.savez_compressed("stego_dataset_val.npz", X=X_val, y=y_val)

print("Numero totale immagini:", len(y))
print("Immagini non stego Training:", (y_train == 0).sum())
print("Immagini stego Training:", (y_train != 0).sum())
print("Immagini non stego Val:", (y_val == 0).sum())
print("Immagini stego Val:", (y_val != 0).sum())
print("Dataset training salvato come 'stego_dataset_train.npz'")
print("Dataset validation salvato come 'stego_dataset_val.npz'")