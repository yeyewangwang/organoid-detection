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

    # get coordinates of input-cell-anchors that actually have an object.
    # shape should be (d, 4) if there are d boxes total
    object_indices = tf.where(tf.sigmoid(y[:,:,:,:,4]) == 1)
    no_object_indices = tf.where(tf.sigmoid(y[:,:,:,:,4]) == 0)

    # bounding box loss:
    # for each GROUND TRUTH BOX,
    # calculate 1-IOU with the corresponding cell's prediction.
    y_object_bb = xywh_to_yxyx(decode_bboxes(y, object_indices, anchors, dims))
    yhat_object_bb = xywh_to_yxyx(decode_bboxes(yhat, object_indices, anchors, dims))
    gl = tfa.losses.GIoULoss()
    iou_loss = tf.cast(gl(y_object_bb, yhat_object_bb), tf.float64)

    # objectness loss where there is an object: logistic objective.
    y_object_to = tf.sigmoid(tf.gather_nd(y, object_indices)[:,4])
    yhat_object_to = tf.sigmoid(tf.gather_nd(yhat, object_indices)[:,4])
    bce = tf.keras.losses.BinaryCrossentropy(from_logits = False)
    objectness_loss = bce(y_object_to, yhat_object_to)

    # objectness loss where there's not an object: logistic objective
    # TODO: if a box has an IOU>.5 with a ground truth box,
    # IGNORE/ do not count it as a false positive
    y_no_object_to = tf.sigmoid(tf.gather_nd(y, no_object_indices)[:,4])
    yhat_no_object_to = tf.sigmoid(tf.gather_nd(yhat, no_object_indices)[:,4])
    no_objectness_loss = bce(y_no_object_to, yhat_no_object_to)
    
    return lambda_coord * iou_loss + objectness_loss + lambda_noobj * no_objectness_loss


# get the intersection over union for two different boxes
# box 1 has format y_min, x_min, y_max, x_max
# box 2 has format y_min, x_min, y_max, x_max
# return float that has the iou for box1 and box2
def calculate_iou(box1, box2):
    #find coordinates for intersecting rectangle
    ymin = max(box1[0], box2[0])
    xmin = max(box1[1], box2[1])
    ymax = min(box1[2], box2[2])
    xmax = min(box1[3], box2[3])

    intersecting_area = max((xmax - xmin) * (ymax - ymin), 0)

    box1_area = (box1[3] - box1[1]) * (box1[2] - box1[0])
    box2_area = (box2[3] - box2[1]) * (box2[2] - box2[0])
    union_area = box1_area + box2_area - intersecting_area

    return intersecting_area / union_area

def bboxes(y, yhat, anchors, dims, threshold=0.5, iou=0.7):
    y = tf.stack(y, axis = 0)
    yhat = tf.cast(yhat, tf.float64)

    object_indices = tf.where(y[:,:,:,:,4] == 1)
    prediction_indices = tf.where(yhat[:,:,:,:,4] > threshold)
    y_object_bb = xywh_to_yxyx(decode_bboxes(y, object_indices, anchors, dims))
    yhat_prediction_bb = xywh_to_yxyx(decode_bboxes(yhat, prediction_indices, anchors, dims))
    return y_object_bb, yhat_prediction_bb

# get the accuracy of how many boxes have been properly predicted
# threshold is for whether our model thinks it's a box
# iou is for whether we consider the iou score enough overlap
# to count as an accurate prediction
def map_and_mse(y, yhat, anchors, dims, threshold = 0.5, iou = 0.7):
    img_width, img_height, grid_dim = dims

    y = tf.stack(y, axis = 0)
    yhat = tf.cast(yhat, tf.float64)

    object_indices = tf.where(tf.sigmoid(y[:,:,:,:,4]) == 1)
    prediction_indices = tf.where(tf.sigmoid(yhat[:,:,:,:,4]) > threshold)

    q_squared_errors = []
    average_precisions = []

    # iterate thru images
    for img in range(y.shape[0]):
        object_img_filter = tf.where(object_indices[:,0,...] == img)
        object_img_indices = tf.gather_nd(object_indices, object_img_filter)
        pred_img_filter = tf.where(prediction_indices[:,0,...] == img)
        pred_img_indices = tf.gather_nd(prediction_indices, pred_img_filter)

        y_object_bb = xywh_to_yxyx(decode_bboxes(y, object_img_indices, anchors, dims))
        yhat_prediction_bb = xywh_to_yxyx(decode_bboxes(yhat, pred_img_indices, anchors, dims))

        curr_se, curr_ap = map_and_mse_single(y_object_bb, yhat_prediction_bb, iou)

        q_squared_errors.append(curr_se)
        average_precisions.append(curr_ap)

    return np.mean(average_precisions), np.mean(q_squared_errors)



def map_and_mse_single(y_object_bb, yhat_prediction_bb, iou = 0.7):
    num_actual_bb = y_object_bb.shape[0]
    num_predict_bb = yhat_prediction_bb.shape[0]

    if num_predict_bb > 1000:
        print("predicting over 1000 boxes :(")

    squared_error = (num_actual_bb - num_predict_bb) ** 2

    if num_predict_bb == 0:
        print("No boxes predicted that met the threshold")
        average_precision = 0.0
    else:
        iou_actual_prediction_matrix = np.empty((num_actual_bb, num_predict_bb))
        for actual_index in range(num_actual_bb):
            for predicted_index in range(num_predict_bb):
                actual_box = y_object_bb[actual_index]
                predicted_box = yhat_prediction_bb[predicted_index]
                curr_iou = calculate_iou(actual_box, predicted_box)
                iou_actual_prediction_matrix[actual_index, predicted_index] = curr_iou

        # find the maximum values of iou in the matrix
        best_predicted_boxes = np.max(iou_actual_prediction_matrix, axis = 1)

        # decide whether it's strong enough iou to be counted -
        # does it meet the iou cutoff set?
        # if there's the same predicted box that is best
        # for multiple actual boxes, we still count it
        count = np.sum(best_predicted_boxes > iou)

        #return the percentage of actual boxes that were predicted
        average_precision = count / num_actual_bb

    return squared_error, average_precision
