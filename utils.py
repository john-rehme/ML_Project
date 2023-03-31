import os
import torch
import torchvision
import torch.nn as nn
from torchvision.io import read_image
# import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib
import matplotlib.pyplot as plt

def read_data(data_path):
    label_to_int = {'NonDemented': 0, 'VeryMildDemented': 1, 'MildDemented': 2, 'ModerateDemented': 3}
    x = []
    y = []
    for folder in os.listdir(data_path)[1:]:
        folder_path = os.path.join(data_path, folder)
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            image = torchvision.io.read_image(file_path)
            x.append(image)
            y.append(folder)
    x = torch.stack(x, 0) / 256
    y = torch.tensor([label_to_int[y] for y in y])
    y_onehot = torch.scatter(torch.zeros(y.shape[0], len(label_to_int)), 1, y.unsqueeze(1), torch.ones(y.shape[0], 1))
    return x, y, y_onehot

def crop(x1, x2, threshold=0):
    print('Cropping images...')
    x = torch.cat((x1, x2))
    rows = x.mean((0, 1, 3))
    cols = x.mean((0, 1, 2))
    top = torch.nonzero(rows > threshold)[0, 0]
    bottom = rows.shape[0] - torch.nonzero(rows > threshold)[-1, 0] - 1
    left = torch.nonzero(cols > threshold)[0, 0]
    right = cols.shape[0] - torch.nonzero(cols > threshold)[-1, 0] - 1
    x_crop = x[:, :, top:-bottom, left:-right]
    print(f'Images were cropped from {x.shape[2]}x{x.shape[3]} to {x_crop.shape[2]}x{x_crop.shape[3]}\n')

    # # below code visualizes changes
    # _, axs = plt.subplots(1, 2, figsize=(10, 5))
    # axs[0].imshow(x[0].squeeze(0), cmap='gray')
    # axs[0].set_title('Original Image')
    # axs[1].imshow(x_crop[0].squeeze(0), cmap='gray')
    # axs[1].set_title('Cropped Image')
    # plt.show()

    return x_crop[:x1.shape[0]], x_crop[x1.shape[0]:]

# def pca_np(x, retained_var): # scrapped
#     U, S, V = np.linalg.svd(x - x.mean(0), full_matrices=False)
#     for k in range(1, S.size + 1):
#         K = k
#         if np.sum(S[:K] ** 2) / np.sum(S ** 2) > retained_var:
#             break
#     return (x - x.mean(0)) @ V.T[:, :K]

def pca(x1, x2, K):
    print('Compressing images with PCA...')
    x = torch.cat((x1, x2))
    x_flat = x.flatten(start_dim=1)
    U, S, V = torch.pca_lowrank(x_flat, q=K, center=True, niter=2)
    z_flat = x_flat @ V
    retained_var = torch.var(z_flat @ V.T) / torch.var(x_flat)
    print(f'{round(retained_var.item() * 100, 4)}% of the variance was retained\n')

    # # below code visualizes changes
    # x_approx_flat = z_flat @ V.T
    # x_approx = x_approx_flat.reshape(x.shape)
    # _, axs = plt.subplots(1, 2, figsize=(10, 5))
    # axs[0].imshow(x[0].squeeze(0), cmap='gray')
    # axs[0].set_title('Original Image')
    # axs[1].imshow(x_approx[0].squeeze(0), cmap='gray')
    # axs[1].set_title(f'PCA Image: k = {K}')
    # plt.show()

    return z_flat[:x1.shape[0]], x_flat[x1.shape[0]:]
    
def calc_loss(logits, y):
    loss_func = nn.CrossEntropyLoss(weight=torch.tensor([2, 2.86, 7.14, 98.48]), reduction='none') # TODO: look into weights parameter
    loss = loss_func(logits, y)
    return loss # shape [BATCH_SIZE]

def calc_accuracy(logits, y):
    y_hat = torch.argmax(logits, 1)
    accuracy = y_hat == y
    return accuracy # shape [BATCH_SIZE]

def calc_cm(logits, y, ):
    y_hat = torch.argmax(logits, 1)
    cm = torch.zeros((logits.shape[1], logits.shape[1]), dtype=int)
    for i, j in zip(y, y_hat):
        cm[i, j] += 1
    return cm

def cm_visualize(cm):
    # matplotlib.use('Agg')
    # ConfusionMatrixDisplay(cm).plot()
    labels = ['     NonDemented', 'VeryMildDemented', '    MildDemented', 'ModerateDemented']
    lbls = ['NonD', 'VMildD', 'MildD', 'ModD']
    print('\nActual\Predicted\t' + '\t'.join(lbls))
    for i, label in enumerate(labels):
        print('{}\t{}'.format(label, '\t'.join(str(cm[i, j].item()) for j in range(len(labels)))))