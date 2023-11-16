
"""
Prompt: "
Transform the data into cppe5\ format. Such as: cppe5["train"] =[0]
{'image_id': 15,
 'image': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=943x663 at 0x7F9EC9E77C10>,
 'width': 943,
 'height': 663,
 'objects': {'id': [114, 115, 116, 117],
  'area': [3796, 1596, 152768, 81002],
  'bbox': [[302.0, 109.0, 73.0, 52.0],
   [810.0, 100.0, 57.0, 28.0],
   [160.0, 31.0, 248.0, 616.0],
   [741.0, 68.0, 202.0, 401.0]],
  'category': [4, 4, 0, 0]}}
"

Prompt: change the image to COCO format.

Since COCO format can be used in DETR, we will convert the data to COCO format.
Prompt: Store the data in metadata.json format rather than cpp5e format. 
The metadata format looks like this:   
{"file_name": "0001.png", "objects": {"bbox": [[302.0, 109.0, 73.0, 52.0]], "categories": [0]}}
{"file_name": "0002.png", "objects": {"bbox": [[810.0, 100.0, 57.0, 28.0]], "categories": [1]}}
{"file_name": "0003.png", "objects": {"bbox": [[160.0, 31.0, 248.0, 616.0], [741.0, 68.0, 202.0, 401.0]], 
"categories": [2, 2]}}

Problem: it hard-coded an arbitrary number of objects for each image.
Prompt: modify the for loop to go through each image_indices, and find bounding boxes that match each index
"""
import pandas as pd
import os
import json
from PIL import Image

# set the dataset type
dataset_type = "train"

# set the directory path
dir_path = f"./{dataset_type}"

# load the bounding box csv file
csv_file = f"./{dataset_type}_bboxes.csv"
bboxes = pd.read_csv(csv_file)
# Find all image inside the dataset directory, assuming name is index
image_indices = [int(image.split(".")[0]) for image in os.listdir(dir_path) if image.endswith(".jpg")]

# initialize the metadata list
metadata = []

category_map = {"organoid": 0}


# loop through each image and its bounding boxes
for index in image_indices:
    # get the image path
    image_path = os.path.join(dir_path, str(index) + ".jpg")
    # open the image
    image = Image.open(image_path)
    # initialize the objects dictionary
    objects = {"bbox": [], "category": []}
    # loop through each bounding box for the image
    for _, row in bboxes[bboxes["image_index"] == index].iterrows():
        # get the bounding box information
        x1, y1, x2, y2 = row[f"x1"], row[f"y1"], row[f"x2"], row[f"y2"]
        bbox = [x1, y1, x2 - x1, y2 - y1]
        category_index = category_map[row[f"class_name"]]
        # add the bounding box information to the objects dictionary
        objects["bbox"].append(bbox)
        objects["category"].append(category_index)
    # create the metadata dictionary for the image
    metadata_image = {
        "file_name": str(index) + ".jpg",
        "image_id": index,
        "objects": objects
    }
    # add the metadata dictionary to the metadata list
    metadata.append(metadata_image)
    # close the image file
    image.close()

# save the metadata list as a JSON Lines file
with open(os.path.join(dir_path, "metadata.jsonl"), "w") as outfile:
    for data in metadata:
        json.dump(data, outfile)
        outfile.write('\n')
