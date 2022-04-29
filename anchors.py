
import numpy as np
import tensorflow as tf
from itertools import chain
import csv
import os
from sklearn.cluster import KMeans
from itertools import chain


# Get list of boxes for generating anchors
def get_boxes(train_labels_dict):
    boxes = list(chain(*train_labels_dict.values))
    boxes_list = list(map(lambda x: x.values(), boxes))
    return np.array(boxes_list)


# Generate anchors using k means
# Ideally this would be done using IOU as a distance metric,
# since Euclidean distance is biased against large boxes.
# however this would mean reimplementing kmeans. Meh!
def generate_anchors(train_labels_dict, num_anchors):
    box_array = get_boxes(train_labels_dict)
    n = box_array.shape[0]

    # get array of widths and heights from train labels
    dims = np.zeros((n, 2))
    # widths
    dims[:,0] = box_array[:,1] - box_array[:,0]
    # heights
    dims[:,1] = box_array[:,3] - box_array[:,2]

    kmeans_model = KMeans(n_clusters = num_anchors)
    kmeans_model.fit(dims)

    # TODO cache results
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


# Assign ground truth to anchor boxes using IOU
# FOR ONE IMAGE
def ground_truth_anchors(box_array, anchors, img_width, img_height, grid_dim = 13):
    num_anchors = anchors.size[0]
    num_boxes = box_array.size[0]
    # TODO: are all images same dims after preprocessing?

    # construct array of all grid coordinates
    cell_width = int(img_width / grid_dim)
    cell_height = int(img_height / grid_dim)
    num_cells = grid_dim * grid_dim
    ints = list(range(grid_dim))
    cell_index = np.array(np.meshgrid(ints, ints)).T.reshape(-1,2)
    
    # coordinates of box centers aka bx and by
    bx = (box_array[:,0] + box_array[:,1] / 2).astype(int)
    by = (box_array[:,2] + box_array[:,3] / 2).astype(int)

    # widths and heights
    bw = box_array[:,1] - box_array[:,0]
    bh = box_array[:,3] - box_array[:,2]

    # assign grid coordinates to boxes aka cx and cy
    grid_x = np.floor_divide(bx, cell_width)
    grid_y = np.floor_divide(by, cell_height)
    cx = grid_x * cell_width
    cy = grid_y * cell_height

    # get offset coordinates aka sigma_tx and sigma_ty
    sigma_tx = np.remainder(bx, cell_width)
    sigma_ty = np.remainder(by, cell_height)

    # get best anchor for each input bbox
    iou = IOU_nocenter(box_array, anchors)
    best_iou_index = np.argmin(iou, axis = 1)
    best_ious = iou[:,best_iou_index]
    
    # anchor dims for each input box
    pw = anchors[best_ious,0]
    ph = anchors[best_ious,1]

    # finally we are ready to calculate "ground truth predictions"
    tx = inverse_sigmoid(bx - cx)
    ty = inverse_sigmoid(by - cy)
    tw = np.log(bw / pw)
    th = np.log(bh / ph)
    to = inverse_sigmoid(iou)

    # if box X has best anchor k at grid coordinate i,j,
    # result[i,j,k] = [tx ty tw th to]
    # somewhat unsure what they should be otherwise but i don't think it matters for loss purposes
    result = np.zeros((num_boxes, grid_dim, grid_dim, num_anchors, 5))
    result[:, grid_x, grid_y, best_iou_index, :] = np.hstack(tx, ty, tw, th, to)

    return result

def inverse_sigmoid(x):
    return -np.log((1 / (x + 1e-8)) - 1)

def load_anchors():
    # TODO
    return
