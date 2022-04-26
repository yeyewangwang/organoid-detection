import csv
import os
import hyperparameters as hp
import numpy as np
import tensorflow as tf
from PIL import Image
from skimage.transform import resize


# Loads in images from a specified directory
# INPUT:
#   path_to_images (str) - path to directory with images
#   augment (boolean) - whether to augment the data
# OUTPUT:
#   img_list (list) - list of images
def get_images(
    path_to_images,
    augment=False
):
    img_dir = os.listdir(path_to_images)
    img_list = []
    for img_path in img_dir:
        img = Image.open(path_to_images + img_path)
        #if we resize, the labels are off
        #resized_img = img.resize(size=(hp.img_height, hp.img_width))

        #if augment:
            # if we wanted to add any augmentation of the data, we could do so here
        
        img_list.append(img)

    return img_list

# Loads in the bounding boxes from csv file
# INPUT:
#   path_to_csv (str) - path to csv file with bounding boxes
# OUTPUT:
#   boxes (dict) - dictionary of image filename to list of bounding boxes
def get_bounding_box_labels(
    path_to_csv,
):
    boxes = {}
    count = 0
    test_substring = "C:/Users/Timothy/Desktop/keras-retinanet/images/test/"
    train_substring = "C:/Users/Timothy/Dropbox/keras-retinanet/images/train/"

    with open(path_to_csv) as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            count += 1

            # Remove the location from the image filenames
            raw_image_name = row['image_path']
            image_name = raw_image_name
            if train_substring in raw_image_name:
                image_name = raw_image_name.replace(train_substring, "")
            elif test_substring in raw_image_name:
                image_name = raw_image_name.replace(test_substring, "")
            
            # Get the coordinates for the bounding box for this organoid
            box_coords = {}
            box_coords['x1'] = row['x1']
            box_coords['x2'] = row['x2']
            box_coords['y1'] = row['y1']
            box_coords['y2'] = row['y2']
            if image_name in boxes:
                boxes[image_name].append(box_coords)
            else:
                boxes[image_name] = [box_coords]

    print("Found " + str(count) + " organoid labels")
    return boxes

# Gets training and testing data
# INPUT:
#   path_to_training_data (str) - path to directory with images for training set
#   path_to_testing_data (str) - path to directory with images for testing set
#   path_to_training_labels (str) - path to csv file with training set labels
#   path_to_testing_labels (str) - path to csv file with testing set labels
#   augment (boolean) - whether to augment the data or not
# OUTPUT:
#   train_images (list) - list of train images
#   train_labels (list) - list of corresponding train labels
#   test_images (list) - list of test images
#   test_labels (list) - list of corresponding test labels
def get_data(
    path_to_training_data,
    path_to_testing_data,
    path_to_training_labels,
    path_to_testing_labels,
    augment=False,
):
    print("Getting training images...")
    train_images = get_images(path_to_training_data, augment)
    print("Found " + str(len(train_images)) + " train images")
    print("Getting testing images...")
    test_images = get_images(path_to_testing_data, augment)
    print("Found " + str(len(test_images)) + " test images")
    print("Getting training labels...")
    train_labels = get_bounding_box_labels(path_to_training_labels)
    print("Found " + str(len(train_labels)) + " train images with organoid labels")
    print("Getting testing labels...")
    test_labels = get_bounding_box_labels(path_to_testing_labels)
    print("Found " + str(len(test_labels)) + " test images with organoid labels")

    return train_images, train_labels, test_images, test_labels

