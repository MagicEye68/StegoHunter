import numpy as np
import torch
import matplotlib.pyplot as plt

class PreprocessLSBLayer(torch.nn.Module):
    def forward(self, x):
        x_uint8 = (x * 255).to(torch.uint8)
        lsb_r = x_uint8[:, 0, :, :] & 1
        lsb_g = x_uint8[:, 1, :, :] & 1
        lsb_b = x_uint8[:, 2, :, :] & 1
        lsb_sum = lsb_r + lsb_g + lsb_b
        return lsb_sum.unsqueeze(1).float() / 3.0

#data = np.load('stego_dataset_train.npz')
data = np.load('stessaFoto.npz')
X = data['X']
y = data['y']

idx_0 = np.where(y == 0)[0][0]
idx_1 = np.where(y == 1)[0][0]

img_0 = X[idx_0]
img_1 = X[idx_1]

imgs = torch.tensor(np.stack([img_0, img_1])).permute(0,3,1,2).float()

preprocess = PreprocessLSBLayer()
imgs_pre = preprocess(imgs)

def tensor_to_numpy_img(tensor):
    return tensor.squeeze().cpu().numpy()


fig, axes = plt.subplots(2, 2, figsize=(10, 8))

axes[0,0].imshow(img_0)
axes[0,0].set_title("Original Image (Non Stego)")
axes[0,0].axis('off')

axes[0,1].imshow(img_1)
axes[0,1].set_title("Original Image (Stego)")
axes[0,1].axis('off')

axes[1,0].imshow(tensor_to_numpy_img(imgs_pre[0]), cmap='gray')
axes[1,0].set_title("Preprocessed LSB (Non Stego)")
axes[1,0].axis('off')

axes[1,1].imshow(tensor_to_numpy_img(imgs_pre[1]), cmap='gray')
axes[1,1].set_title("Preprocessed LSB (Stego)")
axes[1,1].axis('off')

plt.tight_layout()
plt.savefig("img/preprocessed_example.png")
print("Immagine salvata in img/preprocessed_example.png")
