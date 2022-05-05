import keras
import numpy
import numpy as np
import time
import tensorflow as tf
from preprocess import get_data
from anchors import *
from models import run_yolov4, run_one_layer
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

def img_y_to_batch(images_dict, y, batch_size=32):
    """
    Convert images from dictionaries to
    a tensor of (batch size, img height, img width, number of color channels)
    """
    img_list = list(images_dict.values())
    print(f"num of images {len(img_list)}")
    last_i = 0
    for i in range(0, len(img_list), batch_size):
        batch_imgs = np.asarray(img_list[i:i+batch_size])
        yield batch_imgs, y[i:i+batch_size]
        last_i += batch_size
    if last_i < len(img_list):
        yield np.asarray(img_list[last_i:]), y[last_i:]

    # img_array = np.asarray(list(images_dict.values()))
    # return img_array

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
    # TODO: is the 13 a hyperparameter or hard coded?
    dims = (hp.img_width, hp.img_height, hp.grid_dim)
    y = encode_all_bboxes(train_labels, anchors, dims)


    start_time = time.time()
    train_data_gen = img_y_to_batch(train_images, y, hp.batch_size)
    end_time = time.time()
    print(f'Image loading took {end_time - start_time}s.')

    # TODO: un-hardcode training hyperparameters
    lambda_coord = 1
    lambda_noobj = 1

    # train the model
    input = tf.keras.layers.Input([hp.img_height, hp.img_width, 3])
    output = run_yolov4(input)
    simple_output = run_one_layer(input)
    # model = tf.keras.Model(input, output)
    model = tf.keras.Model(input, simple_output)
    model.summary()
    optimizer = tf.keras.optimizers.Adam()

    start_time = time.time()
    for i in range(0):
        loss = []
        print(f"num epochs: {hp.num_epochs}")
        for j, data in enumerate(train_data_gen):
            print(f"num batch {j}")
            img_batch, y_batch = data

            with tf.GradientTape() as tape:
                yhat = model(img_batch, training = True)
                yhat = tf.reshape(yhat, [hp.batch_size, hp.grid_dim, hp.grid_dim, hp.num_anchors, 5])
                curr_loss = yolo_loss(y_batch, yhat, lambda_coord, lambda_noobj, anchors, dims)
                gradients = tape.gradient(curr_loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                # print('WE GOT A TRAINING STEP IN PEOPLE')
            loss.append(curr_loss)
        print(f"epoch = {i} loss = {np.mean(loss)}")
        curr_time = time.time()
        print(f"epoch {i} took {curr_time - start_time}s")
        start_time = curr_time
        #reset the data generator
        train_data_gen = img_y_to_batch(train_images, y, hp.batch_size)

    test_anchors = generate_anchors(test_labels, hp.num_anchors)
    y_test = encode_all_bboxes(test_labels, test_anchors, dims)
    test_data_gen = img_y_to_batch(test_images, y_test, hp.batch_size)
    test_loss = []
    # test the model
    for j, data in enumerate(test_data_gen):
        print(f"Test num batch {j}")
        img_batch, y_batch = data
        yhat = model(img_batch, training=False)
        yhat = tf.reshape(yhat, [-1, hp.grid_dim, hp.grid_dim, hp.num_anchors, 5])
        loss = yolo_loss(y_batch, yhat, lambda_coord, lambda_noobj, test_anchors, dims)
        test_loss.append(loss)
    print(f"Testing loss {np.mean(test_loss)}")

if __name__ == "__main__":
    #parse_args()

    main()
