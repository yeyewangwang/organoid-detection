"""
This Python script is used to find bounding boxes in a dataset 
that stretch beyond the dimensions of their corresponding images.
"""

from data.dataset import get_dataset

def find_stretching_bounding_boxes(image, bounding_boxes):
    image = image.permute(1, 2, 0).numpy() 
    stretching_bounding_boxes = []
    for box in bounding_boxes:
        x1, y1, x2, y2 = box
        if x1 < 0 or y1 < 0 or x2 > image.shape[1] or y2 > image.shape[0]:
            stretching_bounding_boxes.append(box)
    return stretching_bounding_boxes

# Get the dataset
dataset = get_dataset()

# Loop through the dataset
for i, (image, bboxes) in enumerate(dataset):
    stretching_bounding_boxes = find_stretching_bounding_boxes(image, bboxes)
    if len(stretching_bounding_boxes) > 0:
        print("image index:", i)
        print(image.shape)
        print(stretching_bounding_boxes)
