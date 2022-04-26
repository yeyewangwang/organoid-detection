import os
import PIL
import hyperparameters as hp
import numpy as np
import tensorflow as tf
from PIL import Image


# Loads in images from a specified directory
# INPUT:
#   path_to_images (str) - path to directory with images
# OUTPUT:
#   img_list (list) - list of images
def get_images(
    path_to_images,
):
    img_dir = os.listdir(path_to_images)
    img_list = []
    for img_path in img_dir:
        img = Image.open(path_to_images + img_path)
        img_list.append(img.resize((hp.img_height, hp.img_width)))
    return img_list

# Loads in the bounding boxes from csv file
# INPUT:
#   a;sdfk
# OUTPUT:
#   ;alskdjf
def get_bounding_box_labels(
    path_to_csv,
):
    return []

# Gets training and testing data
# INPUT:
#   path_to_training_data (str) - path to directory with images for training set
#   path_to_testing_data (str) - path to directory with images for testing set
#   path_to_training_labels (str) - path to csv file with training set labels
#   path_to_testing_labels (str) - path to csv file with testing set labels
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
):
    print("Getting training images...")
    train_images = get_images(path_to_training_data)
    print("Found " + str(len(train_images)) + " train images")
    print("Getting testing images...")
    test_images = get_images(path_to_testing_data)
    print("Found " + str(len(test_images)) + " test images")
    print("Getting training labels...")
    train_labels = get_bounding_box_labels(path_to_training_labels)
    print("Found " + str(len(train_labels)) + " train labels")
    print("Getting testing labels...")
    test_labels = get_bounding_box_labels(path_to_testing_labels)
    print("Found " + str(len(test_labels)) + " test labels")

    return train_images, train_labels, test_images, test_labels