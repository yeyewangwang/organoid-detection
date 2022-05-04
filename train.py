import numpy as np
from anchors import *
import tensorflow as tf
import tensorflow_addons as tfa


# y and yhat are lists of tensors of shape (grid_dim, grid_dim, num_anchors, 5)
def yolo_loss(y, yhat, lambda_coord, lambda_noobj, anchors, dims):
    img_width, img_height, grid_dim = dims

    # tensorize y
    y = tf.stack(y, axis = 0)
    yhat = tf.cast(yhat, tf.float64)

    # get coordinates of cinput-cell-anchors that actually have an object.
    # shape should be (d, 4) if there are d boxes total
    object_indices = tf.where(y[:,:,:,:,4] == 1)
    no_object_indices = tf.where(y[:,:,:,:,4] == 0)

    # bounding box loss:
    # for each GROUND TRUTH BOX,
    # calculate 1-IOU with the corresponding cell's prediction.
    y_object_bb = xywh_to_yxyx(decode_bboxes(y, object_indices, anchors, dims))
    yhat_object_bb = xywh_to_yxyx(decode_bboxes(yhat, object_indices, anchors, dims))
    gl = tfa.losses.GIoULoss()
    iou_loss = tf.cast(gl(y_object_bb, yhat_object_bb), tf.float64)

    # objectness loss where there is an object: logistic objective.
    y_object_to = tf.gather_nd(y, object_indices)[:,4]
    yhat_object_to = tf.gather_nd(yhat, object_indices)[:,4]
    bce = tf.keras.losses.BinaryCrossentropy(from_logits = False)
    objectness_loss = bce(y_object_to, yhat_object_to)

    # objectness loss where there's not an object: logistic objective
    # TODO: if a box has an IOU>.5 with a ground truth box,
    # IGNORE/ do not count it as a false positive
    y_no_object_to = tf.gather_nd(y, no_object_indices)[:,4]
    yhat_no_object_to = tf.gather_nd(yhat, no_object_indices)[:,4]
    no_objectness_loss = bce(y_no_object_to, yhat_no_object_to)
    
    return lambda_coord * iou_loss + objectness_loss + lambda_noobj * no_objectness_loss

    
