import pandas as pd
import os
import re
import shutil

# set the dataset type
dataset_type = "train"

# set the directory path
dir_path = f"./original_dataset/{dataset_type}/images"

# set the destination directory path
res_path = f"./{dataset_type}"

# get all the files in the directory
files = os.listdir(dir_path)

# load the testing bbox csv file
csv_file = f"./original_dataset/{dataset_type}/{dataset_type}_bboxes.csv"
bboxes= pd.read_csv(csv_file)
bboxes["image_path"] = bboxes["image_path"].apply(lambda x: os.path.basename(x))
# Find the subset index for each bounding box
bboxes["subset"] = bboxes["image_path"].apply(lambda x: int(x.split("_")[1]))
# Find the image index for each bounding box
bboxes["index_in_subset"] = bboxes["image_path"].apply(lambda x: int(x.split(".")[0].split("_")[-1]))
# Initiate a new column for the image index
bboxes["image_index"] = -1

# loop through each file and rename it with a unique index
for i, file in enumerate(sorted(files)):
    print(file)
    # get the file extension
    ext = os.path.splitext(file)[1] 
    # create the new file name with the unique index
    new_name = str(i) + ext
    # copy the file into res_path
    shutil.copy(os.path.join(dir_path, file), os.path.join(res_path, new_name))
    # set the image index in the bounding box directory
    bboxes.loc[bboxes['image_path'] == file, 'image_index'] = i

# drop the image_path column
bboxes = bboxes.drop(columns=["image_path"])

# save the new bounding box csv file
bboxes.to_csv(f"./{dataset_type}_bboxes.csv", index=False)
