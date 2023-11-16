from torch.utils.data import Dataset
import torchvision.transforms as T
import pandas as pd
import cv2
import os

# Set image size and data directory
image_size = 300
DATA_DIR = 'data/train'
BBOXES = 'data/train_bboxes.csv'

# Define image transformations
transform = T.Compose([
        T.ToPILImage(),
        # T.Resize(image_size),
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
def get_dataset():
    return ObjectDetectionDataset(data_dir=DATA_DIR, bboxes_file=BBOXES, transform=transform)
