from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from torchvision import datasets
from skimage import feature
import torch
import torch.nn as nn
from torchvision.utils import make_grid
from torchvision.utils import save_image
# from IPython.display import Image
import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
import random
from models.canny_edge import CannyEdgeDetector
import cv2
import os
import pandas as pd

image_size = 300
DATA_DIR = 'data/train'

transform = T.Compose([
        T.Resize(image_size),
        T.ToTensor()])
dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
batch_size = 64
train_dl = DataLoader(dataset, batch_size, shuffle=False)
# Retrieve the bounding boxes (the correct labels)
train_bboxes = pd.read_csv(os.path.join(DATA_DIR, 'train_bboxes.csv'))
# Find the image index for each bounding box
train_bboxes["image_index"] = train_bboxes["image_path"].apply(lambda x: x.split("/")[-1].split(".")[0])
# Drop redundant columns
train_bboxes = train_bboxes.drop(columns=["image_path", "class_name"])

canny_detector = CannyEdgeDetector()

for inputs, labels in train_dl:
    print(inputs.shape)
    plt.imsave('test.png', inputs[0].permute(1, 2, 0).numpy())
    print(labels.shape)
    image = inputs[0].permute(1, 2, 0).numpy()
    print(image.shape)

    # Bounding boxes
    bboxes = train_bboxes[train_bboxes["image_index"] == 'Subset_1_300x300_001']
    print(bboxes.shape)
    # edges = canny_detector.forward(inputs[0:1].permute(0, 2, 3, 1))

    # Convert to grey images for Canny detector
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = feature.canny(gray_image, sigma=1.0, low_threshold=0.01, high_threshold=0.1)
    print(edges.shape)
    # edges = canny_detector.forward(inputs[0:1])
    # edges_np = edges.squeeze().numpy()

    # Display the original image and the Canny edges side by side
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Original Image
    axes[0].imshow(np.transpose(inputs[0:1].squeeze().numpy(), (1, 2, 0)))
    for i in range(len(bboxes)):
        x, y, x2, y2 = bboxes.iloc[i, :][['x1', 'y1', 'x2', 'y2']].values
        print(x, y, x2, y2)
        w, h = x2 - x, y2 - y
        color = (0, 255, 0)  # Green color for the bounding box
        # Plot the rectangle
        rectangle = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='r', alpha=0.1)
        # Add the Rectangle to the plot
        axes[0].add_patch(rectangle)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # Canny Edges
    axes[1].imshow(edges, cmap='gray')
    axes[1].set_title('Canny Edges')
    axes[1].axis('off')

    plt.show()
    if i == 3:
        break

# def DetectEdges(images: np.ndarray, gaussianSigma: float):

#     edgeImages = []
#     return edgeImages
