import hyperparameters as hp
import keras
import tensorflow as tf
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, BatchNormalization


# Perform convolutional layer
# INPUT:
#   inp (numpy array) - the array to perform the convolution on
#   filt (int) - number of filters
#   kern (int) - kernel size
#   stri (int) - strides
#   pad (string) - padding type
#   act (boolean) - whether to activate
#   bat (boolean) - whether to use batch normalization
# OUTPUT:
#   working_array (numpy array) - resulting numpy array after performing convolutions
def perform_conv(inp, filt, kern, stri, pad, act=True, bat=False):
    working_array = inp

    # maybe add whether to use bias, kernel regularizer, kernel initializer, bias initializer
    working_array = Conv2D(
        filters=filt,
        kernel_size=kern,
        strides=stri,
        padding=pad,
    )(working_array)

    if bat:
        working_array = BatchNormalization()(working_array)
    if act:
        working_array = working_array * tf.math.tanh(tf.math.softplus(working_array))

    return working_array


def perform_residual(inp, filt1, filt2, kern1, kern2, stri, pad, act=True, bat=False):
    """Optional convolutional layers. The input is added to the final results"""
    sc = inp
    working_data = perform_conv(inp=inp, filt=filt1, kern=kern1, stri=stri, pad=pad,
                                act=act, bat=bat)
    working_data = perform_conv(inp=working_data, filt=filt2, kern=kern2, stri=stri, pad=pad,
                                act=act, bat=bat)
    return sc + working_data

def perform_upsample(inp, scale):
    return tf.image.resize(inp, (inp.shape[1] * scale, inp.shape[2] * scale), method='bilinear')

# Run the yolov4 model on the input
# INPUT:
#   inp (numpy array) - input array with data to run model on
# OUTPUT:
#   TBD? out (numpy array) - final array after running model
def run_yolov4(inp):
    working_data = inp

    # Backbone: CSPDarknet53 (53 convolutional layers) used in the yolov4
    # https://github.com/hunglc007/tensorflow-yolov4-tflite/blob/master/core/backbone.py
    working_data = perform_conv(inp=working_data, filt=32, kern=3, stri=1, pad='same')
    working_data = perform_conv(inp=working_data, filt=64, kern=3, stri=2, pad='valid')

    working_data = perform_conv(inp=working_data, filt=32, kern=1, stri=1, pad='same')
    working_data = perform_conv(inp=working_data, filt=64, kern=3, stri=1, pad='same')
    # Note that filt2 in residual has to match with previous filter size
    working_data = perform_residual(inp=working_data, filt1=32, filt2=64,
                                    kern1=1, kern2=3, stri=1, pad='same')  # Residual

    working_data = perform_conv(inp=working_data, filt=128, kern=3, stri=2, pad='valid')

    for i in range(2):
        working_data = perform_conv(inp=working_data, filt=64, kern=1, stri=1, pad='same')
        working_data = perform_conv(inp=working_data, filt=128, kern=3, stri=1, pad='same')
        # NOT SURE HOW TO MATCH WITH PREVIOUS STEP, filt1=64, filt2=64 in cspdarknet
        working_data = perform_residual(inp=working_data, filt1=64, filt2=128,
                                        kern1=1, kern2=3, stri=1, pad='same')  # Residual

    working_data = perform_conv(inp=working_data, filt=256, kern=3, stri=2, pad='valid')

    for i in range(8):
        working_data = perform_conv(inp=working_data, filt=128, kern=1, stri=1, pad='same')
        working_data = perform_conv(inp=working_data, filt=256, kern=3, stri=1, pad='same')
        # NOT SURE HOW TO MATCH WITH PREVIOUS STEP, filt1=64, filt2=64 in cspdarknet
        working_data = perform_residual(inp=working_data, filt1=64, filt2=256,
                                        kern1=1, kern2=3, stri=1, pad='same')  # Residual

    working_data = perform_conv(inp=working_data, filt=512, kern=3, stri=2, pad='valid')

    for i in range(8):
        working_data = perform_conv(inp=working_data, filt=256, kern=1, stri=1, pad='same')
        working_data = perform_conv(inp=working_data, filt=512, kern=3, stri=1, pad='same')
        # NOT SURE HOW TO MATCH WITH PREVIOUS STEP, filt1=256, filt2=256 in cspdarknet
        working_data = perform_residual(inp=working_data, filt1=256, filt2=512,
                                        kern1=1, kern2=3, stri=1, pad='same')  # Residual

    working_data = perform_conv(inp=working_data, filt=1024, kern=3, stri=2, pad='valid')

    for i in range(4):
        working_data = perform_conv(inp=working_data, filt=512, kern=1, stri=1, pad='same')
        working_data = perform_conv(inp=working_data, filt=1024, kern=3, stri=1, pad='same')
        # NOT SURE HOW TO MATCH WITH PREVIOUS STEP, filt1=512, filt2=512 in cspdarknet
        working_data = perform_residual(inp=working_data, filt1=512, filt2=1024,\
                                        kern1=1, kern2=3, stri=1, pad='same')  # Residual

    # GITHUB KEEPS GOING for LAYER 75 and beyond, not sure if this is backbone versus now into neck
    # AVGPOOL
    # CONNECTED
    # SOFTMAX

    # Simplified ending of CSPDarknet53
    working_data = perform_conv(inp=working_data, filt=256, kern=1, stri=1, pad='same')
    l_features = working_data
    # Down sample here
    working_data = perform_conv(inp=working_data, filt=512, kern=3, stri=2, pad='same')
    m_features = working_data
    working_data = perform_conv(inp=working_data, filt=1024, kern=3, stri=2, pad='same')
    s_features = working_data

    # Neck:
    # SSP (increase receptive field and separate most important features from backbone)
    # PANet (feature pyramid network extracting important features from backbone classifier)
    features = s_features
    features = perform_conv(inp=features, filt=256, kern=1, stri=1, pad='same')
    # Upsample by 2 here
    features = perform_upsample(features, 2)
    m_features = perform_conv(inp=m_features, filt=256, kern=1, stri=1, pad='same')
    features = tf.concat([m_features, features], axis=-1)

    for i in range(2):
        features = perform_conv(inp=features, filt=256, kern=1, stri=1, pad='same')
        features = perform_conv(inp=features, filt=512, kern=3, stri=1, pad='same')
    features = perform_conv(inp=features, filt=256, kern=1, stri=1, pad='same')
    m_features = features

    features = perform_conv(inp=features, filt=128, kern=1, stri=1, pad='same')
    # Upsample previous results by 2 here
    features = perform_upsample(features, 2)
    l_features = perform_conv(inp=l_features, filt=128, kern=1, stri=1, pad='same')
    features = tf.concat([l_features, features], axis=-1)
    for i in range(2):
        features = perform_conv(inp=features, filt=128, kern=1, stri=1, pad='same')
        features = perform_conv(inp=features, filt=256, kern=3, stri=1, pad='same')
    features = perform_conv(inp=features, filt=256, kern=1, stri=1, pad='same')
    l_features = features

    # Finish upsampling and concatenating all features together, start predicting bounding boxes
    features = perform_conv(inp=features, filt=256, kern=3, stri=1, pad='same')
    boxes1 = perform_conv(inp=features, filt=3 * 5, kern=1, stri=1,
                   pad='same', act=False)

    features = perform_conv(inp=l_features, filt=256, kern=3, stri=2, pad='same')
    features = tf.concat([features, m_features], axis=-1)
    for i in range(2):
        features = perform_conv(inp=features, filt=256, kern=1, stri=1, pad='same')
        features = perform_conv(inp=features, filt=512, kern=3, stri=1, pad='same')
    features = perform_conv(inp=features, filt=256, kern=1, stri=1, pad='same')
    m_feartures = features

    # Produce the second set of bounding boxes
    features = perform_conv(inp=features, filt=512, kern=3, stri=1, pad='same')
    boxes2 = perform_conv(inp=features, filt=3 * 5, kern=1, stri=1,
                          pad='same', act=False)

    features = perform_conv(inp=m_features, filt=512, kern=3, stri=2, pad='same')
    features = tf.concat([features, s_features], axis=-1)
    for i in range(2):
        features = perform_conv(inp=features, filt=512, kern=1, stri=1, pad='same')
        features = perform_conv(inp=features, filt=1024, kern=3, stri=1, pad='same')
    features = perform_conv(inp=features, filt=512, kern=1, stri=1, pad='same')

    # Produce the third set of bounding boxes
    features = perform_conv(inp=features, filt=1024, kern=3, stri=1, pad='same')
    boxes3 = perform_conv(inp=features, filt=3 * 5, kern=1, stri=1,
                          pad='same', act=False)

    # Heads: YOLOv3
    # Needs to end in a 13x13x10 array

    # Maybe use conv results and produces, small, medium, and large boxes separtately
    # return boxes1, boxes2, boxes3
    return boxes3