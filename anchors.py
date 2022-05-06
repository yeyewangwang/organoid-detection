
import numpy as np
import tensorflow as tf
from itertools import chain
import csv
import os
from sklearn.cluster import KMeans
from itertools import chain
from scipy.special import logit, expit


# Get array of all boxes for generating anchors
def get_boxes(train_labels_dict):
    boxes = list(chain(*train_labels_dict.values()))
    boxes_list = list(map(lambda x: list(x.values()), boxes))
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

    anchors = kmeans_model.cluster_centers_.astype(int)

    return anchors



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

    min_widths = np.minimum.outer(boxes[:,0], anchors[:,0])
    min_heights = np.minimum.outer(boxes[:,1], anchors[:,1])
    intersection = min_widths * min_heights
    
    boxes_areas = boxes[:,0] * boxes[:,1]
    anchors_areas = anchors[:,0] * anchors[:,1]
    union = np.add.outer(boxes_areas, anchors_areas) - intersection

    return intersection / union


# Assign ground truth to anchor boxes using IOU
# FOR ONE IMAGE.
# box_array: shape (N,4) if there are N boxes in the images (x1 x2 y1 y2)
# returns: ground-truth prediction of shape (grid_dim, grid_dim, num_anchors, 5)
def encode_bboxes(box_array, anchors, dims = (300, 300, 13)):
    img_width, img_height, grid_dim = dims
    num_anchors = anchors.shape[0]
    num_boxes = box_array.shape[0]
    # TODO: are all images same dims after preprocessing?

    # construct array of all grid coordinates
    cell_width = int(img_width / grid_dim)
    cell_height = int(img_height / grid_dim)
    num_cells = grid_dim * grid_dim
    ints = list(range(grid_dim))
    cell_index = np.array(np.meshgrid(ints, ints)).T.reshape(-1,2)
    
    # coordinates of box centers aka bx and by
    bx = ((box_array[:,0] + box_array[:,1]) / 2).astype(int)
    by = ((box_array[:,2] + box_array[:,3]) / 2).astype(int)

    # widths and heights
    bw = box_array[:,1] - box_array[:,0]
    bh = box_array[:,3] - box_array[:,2]

    # assign grid coordinates to boxes aka cx and cy
    grid_x = np.minimum(np.floor_divide(bx, cell_width), grid_dim - 1)
    grid_y = np.minimum(np.floor_divide(by, cell_height), grid_dim - 1)
    cx = grid_x * cell_width
    cy = grid_y * cell_height

    # get offset coordinates aka sigma_tx and sigma_ty
    sigma_tx = np.minimum(bx - cx, cell_width - 1)
    sigma_ty = np.minimum(by - cy, cell_height - 1)

    # get best anchor for each input bbox
    iou = IOU_nocenter(box_array, anchors)
    best_iou_index = np.argmax(iou, axis = 1)
    best_ious = iou[list(range(num_boxes)),best_iou_index]
    
    # anchor dims for each input box
    pw = anchors[best_iou_index,0]
    ph = anchors[best_iou_index,1]

    # finally we are ready to calculate "ground truth predictions"
    tx = logit((sigma_tx)/cell_width).reshape(-1,1)
    ty = logit((sigma_ty)/cell_height).reshape(-1,1)
    tw = np.log(bw / pw).reshape(-1,1)
    th = np.log(bh / ph).reshape(-1,1)
    to = np.ones((num_boxes, 1))

    # if box X has best anchor k at grid coordinate i,j,
    # result[i,j,k] = [tx ty tw th to]
    # somewhat unsure what they should be otherwise but i don't think it matters for loss purposes
    preds = np.hstack((tx, ty, tw, th, to))
    result = np.zeros((grid_dim, grid_dim, num_anchors, 5))
    result[grid_x, grid_y, best_iou_index, :] = preds

    # convert to to logit scale
    result[:,:,:,4] = logit(result[:,:,:,4])

    return tf.convert_to_tensor(result)


# return list of tensors of shape (13, 13, 10, 5)
def encode_all_bboxes(train_labels_dict, anchors, dims):
    n = len(train_labels_dict)
    img_width, img_height, grid_dim = dims
    num_anchors = anchors.shape[0]
    out = []

    for box_list in train_labels_dict.values():
        box_array = np.array(list(map(lambda x: list(x.values()), box_list)))
        result = encode_bboxes(box_array, anchors, dims)
        out.append(result)
    
    return out



# Given NN output and INDICES OF CELL/PRIOR COMBOS TO ACTUALLY PREDICT FOR,
# return bounding box predictions for those indices.
# indices should be a tensor of shape (n, 4) --
# each row is a vector of (input,x,y,anchor) for each box
def decode_bboxes(result, indices, anchors, dims = (300, 300, 13)):
    img_width, img_height, grid_dim = dims
    cell_width = int(img_width / grid_dim)
    cell_height = int(img_height / grid_dim)

    t = tf.gather_nd(result, indices)
    grid_x = indices[:,1]
    grid_y = indices[:,2]
    anchor_index = indices[:,3]

    cx = grid_x * cell_width
    cy = grid_y * cell_height
    pw = anchors[anchor_index,0]
    ph = anchors[anchor_index,1]

    return t_to_b(t, cx, cy, pw, ph)

def t_to_b(t, cx, cy, pw, ph):
    tx = t[:,0]
    ty = t[:,1]
    tw = t[:,2]
    th = t[:,3]

    bx = tf.cast(tf.math.sigmoid(tx), tf.int64) + cx
    by = tf.cast(tf.math.sigmoid(ty), tf.int64) + cy
    bw = tf.cast(pw * tf.math.exp(tw), tf.int64) # consider: making this numerically better-conditioned
    bh = tf.cast(ph * tf.math.exp(th), tf.int64)

    return tf.stack([bx, by, bw, bh], axis = 1)

def xywh_to_yxyx(b):
    # y_min, x_min, y_max, x_max
    y_min = b[:,1] - tf.cast(b[:,3] / 2, tf.int64)
    y_max = b[:,1] + tf.cast(b[:,3] / 2, tf.int64)
    x_min = b[:,0] - tf.cast(b[:,2] / 2, tf.int64)
    x_max = b[:,0] + tf.cast(b[:,2] / 2, tf.int64)
    return tf.stack([y_min, x_min, y_max, x_max], axis = 1)
