import cv2
import numpy as np
import sys

def count_lsb_all_channels(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Errore: impossibile caricare l'immagine '{image_path}'")
        return

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    channel_names = ['Rosso', 'Verde', 'Blu']
    total_pixels = img.shape[0] * img.shape[1]

    total_lsb_ones = 0
    total_lsb_zeros = 0

    for i in range(3):
        channel = img[:, :, i]
        lsb_mask = channel & 1
        lsb_ones = np.sum(lsb_mask)
        lsb_zeros = total_pixels - lsb_ones

        total_lsb_ones += lsb_ones
        total_lsb_zeros += lsb_zeros

        print(f"Canale {channel_names[i]}:")
        print(f"  LSB = 1: {lsb_ones} pixel")
        print(f"  LSB = 0: {lsb_zeros} pixel")
        print()

    print(f"Totale pixel per canale: {total_pixels}")
    print(f"\n=== Totali complessivi ===")
    print(f"LSB = 1 su tutti i canali: {total_lsb_ones}")
    print(f"LSB = 0 su tutti i canali: {total_lsb_zeros}")
    print(f"Somma LSB (1+0): {total_lsb_ones + total_lsb_zeros}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python3 lsb.py <percorso_immagine>")
    else:
        count_lsb_all_channels(sys.argv[1])
