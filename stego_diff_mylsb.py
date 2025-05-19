import matplotlib.pyplot as plt
import numpy as np
import cv2

def extract_message_lsb(image: np.ndarray) -> str:
    flat_image = image.reshape(-1, 3)
    bits = []

    for pixel in flat_image:
        for channel in pixel:
            bit = channel & 1
            bits.append(str(bit))
    message = ""
    current_count = 0

    for bit in bits:
        if bit == '1':
            current_count += 1
        else:
            if current_count > 0:
                if 1 <= current_count <= 26:
                    message += chr(ord('a') + current_count - 1)
                elif current_count == 27:
                    message += ','
                elif current_count == 28:
                    message += '.'
                    return message
                elif current_count == 29:
                    message += '?'
                elif current_count == 30:
                    message += '!'
                else:
                    message += ' '
                current_count = 0
            else:
                message += ' '

    if current_count > 0:
        if 1 <= current_count <= 26:
            message += chr(ord('a') + current_count - 1)
        elif current_count == 27:
            message += ','
        elif current_count == 28:
            message += '.'
            return message
        elif current_count == 29:
            message += '?'
        elif current_count == 30:
            message += '!'
        else:
            message += ' '

    return message


clean = cv2.imread("img/clean_sample.png")
stego = cv2.imread("img/stego_sample.png")

clean_rgb = cv2.cvtColor(clean, cv2.COLOR_BGR2RGB)
stego_rgb = cv2.cvtColor(stego, cv2.COLOR_BGR2RGB)

diff = cv2.absdiff(clean_rgb, stego_rgb)

diff_amp = diff * 20

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.title("Originale")
plt.imshow(clean_rgb)
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Steganografata")
plt.imshow(stego_rgb)
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Pixel Diff")
plt.imshow(diff_amp)
plt.axis("off")

plt.tight_layout()
plt.savefig("img/diff_visualization_mylsb.png")
print("Immagine salvata come diff_visualization_mylsb.png")

hidden_message = extract_message_lsb(stego_rgb)
print("Messaggio estratto:", hidden_message)
