from torch.utils.data import Dataset
import torchvision.transforms as T
from skimage import feature
import matplotlib.pyplot as plt
from matplotlib import patches
import cv2
import os
import pandas as pd

# Set image size and data directory
image_size = 300
DATA_DIR = 'data/train'
BBOXES = 'data/train_bboxes.csv'

# Define image transformations
transform = T.Compose([
        T.ToPILImage(),
        T.Resize(image_size),
        T.ToTensor()])

class ObjectDetectionDataset(Dataset):
    """
    A PyTorch Dataset class for object detection.

    Args:
        data_dir (str): The directory containing the images.
        bboxes_file (str): The file containing the bounding boxes for the images.
        transform (callable, optional): A function/transform that takes in an PIL image and returns a transformed version.
    """
    def __init__(self, data_dir, bboxes_file, transform=None):
        self.data_dir = data_dir
        self.bboxes = pd.read_csv(bboxes_file)
        self.transform = transform

    def __len__(self):
        return len(os.listdir(self.data_dir))

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, f"{idx}.jpg")
        image = cv2.imread(img_path)
        bboxes = self.bboxes[self.bboxes["image_index"] == idx][["x1", "y1", "x2", "y2"]].values
        if self.transform:
            image = self.transform(image)
        return image, bboxes

# Create the dataset
dataset = ObjectDetectionDataset(data_dir=DATA_DIR, bboxes_file=BBOXES, transform=transform)

# Loop through the dataset
for i, (image, bboxes) in enumerate(dataset):
    # Convert image to numpy array
    image = image.permute(1, 2, 0).numpy()    

    # Convert image to grayscale for Canny detector
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect edges using Canny detector
    edges = feature.canny(gray_image, sigma=1.0, low_threshold=0.01, high_threshold=0.1)
    
    # Display the original image and the Canny edges side by side
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    # Original Image
    axes[0].imshow(image)
    for j in range(len(bboxes)):
        x, y, x2, y2 = bboxes[j]
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
    
    # Break loop after 100 iterations
    if i == 100:
        break
