import keras
import numpy
import numpy as np
import time
import tensorflow as tf
from preprocess import get_data
from anchors import *
from models import run_yolov4
import hyperparameters as hp
from train import yolo_loss

def parse_args():
    """ Perform command-line argument parsing. """

    parser = argparse.ArgumentParser(
        description="Adding argument parser",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--an_argument',
        required=True,
        choices=['1', '3'],
        help='''TBD''')

    return parser.parse_args()

def images_dict_to_batch(images_dict):
    """
    Convert images from dictionaries to
    a tensor of (batch size, img height, img width, number of color channels)
    """
    img_array = np.asarray(list(images_dict.values()))
    print("Input dimensions:", img_array.shape)
    return img_array

def main():
    data_dir = "data"
    print("Getting data...")
    train_images, train_labels, test_images, test_labels = get_data(
        path_to_training_data=data_dir + "/train/",
        path_to_testing_data=data_dir + "/test/",
        path_to_testing_labels=data_dir + "/test_labels.csv",
        path_to_training_labels=data_dir + "/train_labels.csv",
    )
    print("Data retrieved!")
    #print("Checking that images have been resized...")
    #print((300,300) == train_images[0].size)
    
    
    # get ground truth outputs
    anchors = generate_anchors(train_labels, hp.num_anchors)
    # TODO: are these dimensions hyperparameters or hard coded?
    dims = (hp.img_height, hp.img_width, 13)
    y = encode_all_bboxes(train_labels, anchors, dims)

    start_time = time.time()
    train_images = images_dict_to_batch(train_images)
    test_images = images_dict_to_batch(test_images)
    end_time = time.time()
    print(f'Image loading took {end_time - start_time}s.')

    # TODO: un-hardcode training hyperparameters
    lambda_coord = 1
    lambda_noobj = 1

    # train the model
    input = tf.keras.layers.Input([hp.img_height, hp.img_width, 3])
    output = run_yolov4(input)
    model = tf.keras.Model(input, output)
    # model.summary()
    optimizer = tf.keras.optimizers.Adam()

    for i in range(2):
        with tf.GradientTape() as tape:
            yhat = model(train_images, training = True)
            print(yhat.shape)
            loss = yolo_loss(y, yhat, lambda_coord, lambda_noobj, anchors, dims)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # test the model
    yhat = model(test_images, training=False)
    loss = yolo_loss(y, yhat, hp.lambda_coord, hp.lambda_noobj, anchors, dims)
    print(f"Testing loss {loss}")

if __name__ == "__main__":
    #parse_args()

    main()
