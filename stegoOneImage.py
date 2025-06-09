import numpy as np
import cv2
import random
from PIL import Image
from faker import Faker
import re
import numpy as np
import os
from PIL import Image


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

def generate_random_sentence(max_length=3072):
    while True:
        sentence = fake.sentence(nb_words=30)
        cleaned = clean_message(sentence)
        encoded = classicLSB(cleaned)
        if len(encoded) <= max_length:
            return cleaned

def clean_message(msg):
    msg = msg.lower()
    allowed = re.compile(r"[a-z ,.?!)]+")
    cleaned = ''.join(ch for ch in msg if allowed.match(ch))
    return cleaned

def classicLSB(message:str) -> str:
    return ''.join(format(ord(char), '08b') for char in message)

def stego_random_method(image_path: str, output_path: str):
    img_pil = Image.open(image_path).convert("RGB")
    img_pil = img_pil.resize((128, 128), Image.Resampling.LANCZOS)
    img_np = np.array(img_pil)

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

    method, channel = random.choice(stego_methods)
    print(f"Metodo scelto: {method} - Canale: {channel}")

    MESSAGE = generate_random_sentence()

    if method == 'rgb':
        img_stego = hide_message_lsb_sequential(img_np, message=MESSAGE, rgb=True)
    elif method == 'single':
        img_stego = hide_message_lsb_sequential(img_np, message=MESSAGE, rgb=False, channel=channel)
    elif method == 'rgb_vertical':
        img_stego = hide_message_lsb_sequential_vertical(img_np, message=MESSAGE, rgb=True)
    elif method == 'single_vertical':
        img_stego = hide_message_lsb_sequential_vertical(img_np, message=MESSAGE, rgb=False, channel=channel)
    else:
        raise ValueError(f"Metodo stego sconosciuto: {method}")

    stego_bgr = cv2.cvtColor(img_stego, cv2.COLOR_RGB2BGR)

    cv2.imwrite(output_path, stego_bgr)
    print(f"Immagine stego salvata in: {output_path}")

def stego_all_methods(image_path: str, output_folder: str):
    img_pil = Image.open(image_path).convert("RGB")
    img_pil = img_pil.resize((128, 128), Image.Resampling.LANCZOS)
    img_np = np.array(img_pil)

    os.makedirs(output_folder, exist_ok=True)

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

    for method, channel in stego_methods:
        print(f"Metodo in esecuzione: {method} - Canale: {channel}")

        MESSAGE = generate_random_sentence()

        if method == 'rgb':
            img_stego = hide_message_lsb_sequential(img_np, message=MESSAGE, rgb=True)
        elif method == 'single':
            img_stego = hide_message_lsb_sequential(img_np, message=MESSAGE, rgb=False, channel=channel)
        elif method == 'rgb_vertical':
            img_stego = hide_message_lsb_sequential_vertical(img_np, message=MESSAGE, rgb=True)
        elif method == 'single_vertical':
            img_stego = hide_message_lsb_sequential_vertical(img_np, message=MESSAGE, rgb=False, channel=channel)
        else:
            raise ValueError(f"Metodo stego sconosciuto: {method}")

        stego_bgr = cv2.cvtColor(img_stego, cv2.COLOR_RGB2BGR)

        filename = f"{method}"
        if channel:
            filename += f"_{channel}"
        filename += "_stego.png"

        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, stego_bgr)
        print(f"Immagine stego salvata in: {output_path}")

fake = Faker()
stego_random_method("img/clean_sample.png", "img/stegoimg.png")
#stego_all_methods("img/clean_sample.png", "output_stego")
