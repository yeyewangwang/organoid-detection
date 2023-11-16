
from skimage import feature
import matplotlib.pyplot as plt
from matplotlib import patches
import cv2
import os
from data.dataset import get_dataset

# Get the dataset
dataset = get_dataset()

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
