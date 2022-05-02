import hyperparameters as hp
import keras
import tensorflow as tf
from keras.models import Sequential
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
def perform_conv(inp, filt, kern, stri, pad, act=True, bat=True):
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


# Run the yolov4 model on the input
# INPUT:
#   inp (numpy array) - input array with data to run model on
# OUTPUT:
#   TBD? out (numpy array) - final array after running model
def run_yolov4(inp):
    working_data = keras.Input(shape=(hp.img_height, hp.img_width))(inp)

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
        working_data = perform_residual(inp=working_data, filt1=512, filt2=1024,
                                        kern1=1, kern2=3, stri=1, pad='same')  # Residual

    # GITHUB KEEPS GOING for LAYER 75 and beyond, not sure if this is backbone versus now into neck
    # AVGPOOL
    # CONNECTED
    # SOFTMAX

    # Neck:
    # SSP (increase receptive field and separate most important features from backbone)
    # PANet (feature pyramid network extracting important features from backbone classifier)

    # Heads: YOLOv3
    # Needs to end in a 13x13x10 array

    # Maybe use conv results and produces, small, medium, and large boxes separtately
