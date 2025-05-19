from torchvision import datasets, transforms
from sklearn.utils import shuffle
from faker import Faker
from tqdm import tqdm
import numpy as np
import torch
import cv2
import re

transform = transforms.ToTensor()
dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset))
images, labels = next(iter(loader))
images_np = (images.numpy() * 255).astype(np.uint8).transpose(0, 2, 3, 1)

fake = Faker()

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
    message += '\0'
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

def hide_message_lsb(image: np.ndarray, message: str) -> np.ndarray:

    #binary = classicLSB(message)
    binary = myLSB(message)
    image = image.copy()
    h, w, c = image.shape

    if c < 3:
        raise ValueError("image_channel < 3")

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

def generate_stego_dataset(images):
    images = np.clip(images * 255, 0, 255).astype(np.uint8)
    processed_images = []
    labels = []
    stessaFoto = []
    stessaFotoLabels = []

    for i, img in enumerate(tqdm(images, desc="Generazione dataset")):
        if i == 99:
            cv2.imwrite("img/clean_sample.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            stessaFoto.append(img)
            stessaFotoLabels.append(0)
            MESSAGE = generate_random_sentence()
            img = hide_message_lsb(img, message=MESSAGE)
            cv2.imwrite("img/stego_sample.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            stessaFoto.append(img)
            stessaFotoLabels.append(1)
            XFotoUnica=np.array(stessaFoto, dtype=np.float32) / 255.0
            YFotoUnica=np.array(stessaFotoLabels)
            np.savez_compressed("stessaFoto.npz", X=XFotoUnica, y=YFotoUnica)

        if i % 2 == 0:
            MESSAGE = generate_random_sentence()
            img = hide_message_lsb(img, message=MESSAGE)
            labels.append(1)
        else:
            labels.append(0)

        processed_images.append(img)

    X = np.array(processed_images, dtype=np.float32) / 255.0
    y = np.array(labels)

    return X, y

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