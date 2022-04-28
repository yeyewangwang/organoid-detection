
import numpy as np
import tensorflow as tf
from itertools import chain
import csv
import os
from sklearn.cluster import KMeans

# TODO: Rewrite this to take in PREPROCESSED labels.
def get_boxes(path_to_csv):
    test_substring = "C:/Users/Timothy/Desktop/keras-retinanet/images/test/"
    train_substring = "C:/Users/Timothy/Dropbox/keras-retinanet/images/train/"

    rows = []

    with open(path_to_csv) as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            count += 1
            rows.append([int(row['x1']), int(row['x2']), int(row['y1']), int(row['y2'])])

    return np.array(rows)


# Generate anchors using k means
# Ideally this would be done using IOU as a distance metric,
# since Euclidean distance is biased against large boxes.
# however this would mean reimplementing kmeans. Meh!
def generate_anchors(box_array, num_anchors):
    n = box_array.shape[0]

    # get array of widths and heights from train labels
    dims = np.zeros((n, 2))
    # widths
    dims[:,0] = box_array[:,1] - box_array[:,0]
    # heights
    dims[:,1] = box_array[:,3] - box_array[:,2]

    kmeans_model = KMeans(n_clusters = num_anchors)
    kmeans_model.fit(dims)
    return kmeans_model.cluster_centers_

# Compute IOU of all boxes against all anchors. ASSUME ANCHORS CENTERED AT BOX CENTER.
# boxes1: shape (N, 4)
# anchors: shape (M, 2)
# returns: shape (N, M) which shows IOU for box N and anchor M
def IOU_nocenter(box_array, anchors):

    n = box_array.shape[0]
    m = anchors.shape[0]

    # get array of widths and heights from train labels
    boxes = np.zeros((n, 2))
    # widths
    boxes[:,0] = box_array[:,1] - box_array[:,0]
    # heights
    boxes[:,1] = box_array[:,3] - box_array[:,2]


    min_widths = np.min.outer(boxes[:,0], anchors[:,0])
    min_heights = np.min.outer(boxes[:,1], anchors[:,1])
    intersection = min_widths * min_heights
    
    boxes_areas = boxes[:,0] * boxes[:,1]
    anchors_areas = anchors[:,0] * anchors[:,1]
    union = np.add.outer(boxes_areas, anchors_areas) - intersection

    return intersection / union

# Compute IOU of all boxes against all anchors. DO NOT ASSUME ANCHORS CENTERED AT BOX CENTER.
# box_array: shape (N, 4)
# anchors: shape (M, 4)
# returns: shape (N, M) which shows IOU for box N and anchor M 
def IOU(box_array, anchor_array):
    # TODO. probably user tensorflow.


    return

# Assign ground truth to anchor boxes using IOU
def ground_truth_anchors(box_array, img_width, img_height, centers_sqrt = 13, num_anchors = 10):

    # TODO: are all images same dims after preprocessing?

    # given centers_sqrt, construct array of all grid coordinates and anchor indices
    grid_el_width = int(img_width / centers_sqrt)
    grid_el_height = int(img_height / centers_sqrt)
    centers = centers_sqrt * centers_sqrt
    ints = list(range(centers_sqrt))
    center_index = np.array(np.meshgrid(ints, ints)).T.reshape(-1,2)

    # translate grid coordinate into pixel coordinate of image
    center_coords = np.copy(center_index)
    center_coords[:,0] = center_coords[:,0] * grid_el_width + int(grid_el_width/2)
    center_coords[:,1] = center_coords[:,1] * grid_el_height + int(grid_el_height/2)

    # get anchor w/h
    anchors = generate_anchors(box_array, num_anchors)

    # make array of all anchor-center combos
    anchor_array = np.zeros((centers * num_anchors, 4))

    for i in range(num_anchors):
        first = i * centers
        last = (i + 2) * centers
        curr_anchor_width = anchors[i,0]
        curr_anchor_height = anchors[i,1]
        # x1
        anchor_array[first:last, 0] = center_coords[:,0] - int(curr_anchor_width/2)
        # x2
        anchor_array[first:last, 1] = anchor_array[first:last, 0] + curr_anchor_width
        # y1
        anchor_array[first:last, 2] = center_coords[:,1] - int(curr_anchor_height/2)
        # y2
        anchor_array[first:last, 3] = anchor_array[first:last, 1] + curr_anchor_height

    iou = IOU(box_array, anchor_array)

    # get best anchor-center combo for each input bbox
    best_iou_index = np.argmin(iou, axis = 1)

    # TODO: not sure what to return...

    return  best_iou_index


