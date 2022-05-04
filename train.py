import numpy as np
from anchors import *
import tensorflow as tf
import tensorflow_addons as tfa

def yolo_loss(y, yhat, lambda_coord, lambda_noobj, anchors, dims):
    # TODO: sum up yolo_loss_single for each element in y, yhat
    return 0

# TODO convert to tensor operations
# y and yhat have shape (grid_dim, grid_dim, num_anchors, 5)
def yolo_loss_single(y, yhat, lambda_coord, lambda_noobj, anchors, dims):
    img_width, img_height, grid_dim = dims
    # get coordinates of cell-anchors that actually have an object.
    # shape should be (n, 3)
    object_indices = tf.where(y[:,:,:,4] == 1)
    no_object_indices = tf.where(y[:,:,:,4] == 0)

    # bounding box loss:
    # for each GROUND TRUTH BOX,
    # calculate 1-IOU with the corresponding cell's prediction.
    y_object_bb = xywh_to_yxyx(decode_bboxes(y, object_indices, anchors, dims))
    yhat_object_bb = xywh_to_yxyx(decode_bboxes(yhat, object_indices, anchors, dims))
    gl = tfa.losses.GIoULoss()
    iou_loss = gl(y_object_bb, yhat_object_bb)  


    # objectness loss where there is an object: logistic objective.
    y_object_to = tf.gather_nd(y, object_indices)[:,4]
    yhat_object_to = tf.gather_nd(yhat, object_indices)[:,4]
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    objectness_loss = bce(y_object_to, yhat_object_to)

    # objectness loss where there's not an object: logistic objective
    # TODO: if a box has an IOU>.5 with a ground truth box,
    # IGNORE/ do not count it as a false positive
    y_no_object_to = tf.gather_nd(y, no_object_indices)[:,4]
    yhat_no_object_to = tf.gather_nd(yhat, no_object_indices)[:,4]
    no_objectness_loss = bce(y_object_to, yhat_object_to)
    
    return lambda_coord * iou_loss + objectness_loss + lambda_noobj * no_objectness_loss

    
