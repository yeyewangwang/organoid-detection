import csv
import os
import hyperparameters as hp
import numpy as np
import tensorflow as tf
from PIL import Image
from skimage.transform import rescale


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
    scales_list = []
    for img_path in img_dir:
        img = np.asarray(Image.open(path_to_images + img_path))
        #if we resize, the labels are off
        if img.shape[0] != hp.img_height:
            scale_factor = hp.img_height / img.shape[0]
            img = rescale(img, scale=scale_factor, channel_axis=2)
        else:
            scale_factor = 1

        #if augment:
            # if we wanted to add any augmentation of the data, we could do so here
        
        img_list.append(img)
        scales_list.append(scale_factor)

    return img_list, scales_list

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
            box_coords['x1'] = int(row['x1'])
            box_coords['x2'] = int(row['x2'])
            box_coords['y1'] = int(row['y1'])
            box_coords['y2'] = int(row['y2'])
            if image_name in boxes:
                boxes[image_name].append(box_coords)
            else:
                boxes[image_name] = [box_coords]

    print("Found " + str(count) + " organoid labels")
    return boxes


def find_images_by_boxes(boxes, path_to_images, augment=False):
    """
    Take a dictionary of boxes, where each key is an image file name. Return each image,
    rescale the image, and rescale the box coordinates by the same scale.
    """
    images = {}
    for img_path, img_boxes in boxes.items():
        img = np.asarray(Image.open(path_to_images + img_path))
        # if we resize, the labels are off
        if img.shape[0] != hp.img_height:
            scale_factor = hp.img_height / img.shape[0]
            img = rescale(img, scale=scale_factor, channel_axis=2)
        else:
            scale_factor = 1

        for box in img_boxes:
            for coord in box:
                box[coord] *= scale_factor
        images[img_path] = img

        # Add augmentation here, if necessary.
        # if augment:
        #     pass
    return images, boxes


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
    print("Getting training labels...")
    train_labels = get_bounding_box_labels(path_to_training_labels)
    print("Found " + str(len(train_labels)) + " train images with organoid labels")
    print("Getting testing labels...")
    test_labels = get_bounding_box_labels(path_to_testing_labels)
    print("Found " + str(len(test_labels)) + " test images with organoid labels")

    print("Finding training images by the file names given in the labels")
    train_images, train_labels  = find_images_by_boxes(train_labels, path_to_training_data, augment=augment)
    print("Finding testing images by the file names given in the labels")
    test_images, test_labels = find_images_by_boxes(test_labels, path_to_testing_data, augment=augment)

    return train_images, train_labels, test_images, test_labels
