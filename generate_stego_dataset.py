from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from sklearn.utils import shuffle
from faker import Faker
from PIL import Image
from tqdm import tqdm
import numpy as np
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
            return None  # indica che è saltata

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

    images_np = np.stack(images).astype(np.uint8)  # (N, H, W, 3)

    # Converti in tensore PyTorch float normalizzato e permuta dimensioni in (N, 3, H, W)
    images_tensor = torch.tensor(images_np / 255., dtype=torch.float32).permute(0, 3, 1, 2)

    return images_tensor

def load_cifar10_numpy(train=True, download=True, root='./data') -> 'np.ndarray':
    transform = transforms.ToTensor()
    dataset = datasets.CIFAR10(root=root, train=train, download=download, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset))
    images, _ = next(iter(loader))
    images_np = (images.numpy() * 255).astype('uint8').transpose(0, 2, 3, 1)
    return images_np

def generate_random_sentence(max_length=3072):
    while True:
        sentence = fake.sentence(nb_words=5)
        cleaned = clean_message(sentence)
        encoded = myLSB(cleaned)
        if len(encoded) <= max_length:
            return cleaned

def clean_message(msg):
    msg = msg.lower()
    allowed = re.compile(r"[a-z ,.?!)]+")
    cleaned = ''.join(ch for ch in msg if allowed.match(ch))
    return cleaned

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

def hide_message_lsb(image: np.ndarray, message: str, rgb:bool, channel: str = None) -> np.ndarray:

    binary = classicLSB(message)
    #binary = myLSB(message)
    image = image.copy()
    h, w, c = image.shape

    if c < 3:
        raise ValueError("image_channel < 3")
    if(rgb):
        flat_image = image.reshape(-1, 3)
        total_channels = flat_image.shape[0] * 3

        if len(binary) > total_channels:
            raise ValueError("len(message) > channels")

        for i in range(len(binary)):
            pixel_idx = i // 3
            channel_idx = i % 3
            flat_image[pixel_idx][channel_idx] &= 0xFE
            flat_image[pixel_idx][channel_idx] |= int(binary[i]) #bitwise OR

        stego_image = flat_image.reshape((h, w, 3)).astype(np.uint8)
        return stego_image
    else:
        # Mappa il canale scelto a un indice (r=0, g=1, b=2)
        channel = channel.lower()
        channel_map = {'r': 0, 'g': 1, 'b': 2}

        ch_idx = channel_map[channel]

        flat_image = image.reshape(-1, 3)
        total_pixels = flat_image.shape[0]

        if len(binary) > total_pixels:
            raise ValueError(f"len(message) > {channel} channel capacity")

        for i in range(len(binary)):
            flat_image[i][ch_idx] &= 0xFE          # clear LSB del canale scelto
            flat_image[i][ch_idx] |= int(binary[i]) # set nuovo bit

        stego_image = flat_image.reshape((h, w, 3)).astype(np.uint8)
        return stego_image


def generate_stego_dataset(images):
    # Se è torch.Tensor, convertilo in NumPy
    if isinstance(images, torch.Tensor):
        images = images.permute(0, 2, 3, 1).numpy()  # (N, 3, H, W) → (N, H, W, 3)

    # Se ha shape (N, 3, H, W) in numpy (non torch), trasponi
    if images.ndim == 4 and images.shape[1] == 3 and images.shape[-1] != 3:
        images = images.transpose(0, 2, 3, 1)  # (N, 3, H, W) → (N, H, W, 3)

    # Clip e converti a uint8
    images = np.clip(images * 255, 0, 255).astype(np.uint8)
    processed_images = []
    labels = []
    stessaFoto = []
    stessaFotoLabels = []

    stego_methods = [
        ('rgb', None),     # rgb=True, canale ignorato
        ('single', 'r'),   # rgb=False, canale 'r'
        ('single', 'g'),   # rgb=False, canale 'g'
        ('single', 'b'),   # rgb=False, canale 'b'
    ]

    for i, img in enumerate(tqdm(images, desc="Generazione dataset")):
        if i == 99:
            cv2.imwrite("img/clean_sample.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            stessaFoto.append(img)
            stessaFotoLabels.append(0)
            MESSAGE = generate_random_sentence()
            img = hide_message_lsb(img, message=MESSAGE, rgb=True)
            cv2.imwrite("img/stego_sample.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            stessaFoto.append(img)
            stessaFotoLabels.append(1)
            XFotoUnica=np.array(stessaFoto, dtype=np.float32) / 255.0
            YFotoUnica=np.array(stessaFotoLabels)
            np.savez_compressed("stessaFoto.npz", X=XFotoUnica, y=YFotoUnica)

        label_type = i % 5

        if label_type == 0:
            # immagine pulita
            labels.append(0)
            processed_images.append(img)
        else:
            # stego, scegli la tecnica corrispondente a label_type-1
            method, channel = stego_methods[label_type - 1]

            MESSAGE = generate_random_sentence()
            if method == 'rgb':
                img_stego = hide_message_lsb(img, message=MESSAGE, rgb=True)
            else:
                img_stego = hide_message_lsb(img, message=MESSAGE, rgb=False, channel=channel)

            labels.append(label_type)
            processed_images.append(img_stego)

    X = np.array(processed_images, dtype=np.float32) / 255.0
    y = np.array(labels)
    if len(processed_images) != len(labels):
     print(f"[ERRORE] Numero immagini ({len(processed_images)}) diverso da numero etichette ({len(labels)})")
    return X, y

#http://dde.binghamton.edu/download/ImageDB/BOSSbase_1.01.zip
#images_np = load_bossbase_numpy("./BOSSbase_1.01", image_size=128)
images_np = load_cifar10_numpy()
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
print("Immagini stego Training:", (y_train == 1).sum())
print("Immagini non stego Val:", (y_val == 0).sum())
print("Immagini stego Val:", (y_val == 1).sum())
print("Dataset training salvato come 'stego_dataset_train.npz'")
print("Dataset validation salvato come 'stego_dataset_val.npz'")