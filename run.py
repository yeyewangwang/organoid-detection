import hyperparameters as hp
import numpy as np
import time
import tensorflow as tf
from pathlib import Path

from anchors import *
from models import run_yolov4, run_one_layer
from preprocess import get_data
from train import *


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

def main(saved_weights_path="saved_weights/new_experiment",
         save_per_epoch=False,
         retrain=True,
         eval_train=True,
         test_only=False):
    data_dir = "data"
    print("Getting data...")
    train_images, train_labels, test_images, test_labels = get_data(
        path_to_training_data=data_dir + "/train/",
        path_to_testing_data=data_dir + "/test/",
        path_to_testing_labels=data_dir + "/test_labels.csv",
        path_to_training_labels=data_dir + "/train_labels.csv",
    )
    print("Data retrieved!")


    # HYPERPARAMETER SETTING
    input = tf.keras.layers.Input([hp.img_height, hp.img_width, 3])
    output = run_yolov4(input)

    # Testing for just the training, testing, and loss on a simplified model
    #output = run_one_layer(input)

    print(output.shape)
    
    #Set a parameter to use for the grid dim based on the model shape instead of hardcoding it
    grid_dim = output.shape[2]

    model = tf.keras.Model(input, output)

    num_epochs = hp.num_epochs
    if retrain:
        print(f"Training from scratch")
    elif test_only:
        num_epochs = 0
        model.load_weights(saved_weights_path)
        print(f"Restoring weights from {saved_weights_path}")
    else:
        model.load_weights(saved_weights_path)
        print("Training from saved weights")
        print(f"Restoring weights from {saved_weights_path}")

    print("Getting ground truth outputs...")
    anchors = generate_anchors(train_labels, hp.num_anchors)
    dims = (hp.img_width, hp.img_height, grid_dim)
    y = encode_all_bboxes(train_labels, anchors, dims)
    start_time = time.time()
    train_data_gen = img_y_to_batch(train_images, y, hp.batch_size)
    end_time = time.time()
    print(f'Image loading took {end_time - start_time}s.')

    optimizer = tf.keras.optimizers.Adam()

    # train the model
    if not test_only:
        model.summary()
        print("Now training the model...")
    start_time = time.time()
    for i in range(num_epochs):
        loss = []
        print(f"num epochs: {hp.num_epochs}")
        for j, data in enumerate(train_data_gen):
            # if j == 2:
            #     break

            print(f"num batch {j}")
            img_batch, y_batch = data

            with tf.GradientTape() as tape:
                yhat = model(img_batch, training = True)
                yhat = tf.reshape(yhat, [hp.batch_size, grid_dim, grid_dim, hp.num_anchors, 5])
                curr_loss = yolo_loss(y_batch, yhat, hp.lambda_coord, hp.lambda_noobj, anchors, dims)
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

        # Save weights either per epoch or if last epoch
        if save_per_epoch or i == num_epochs - 1:
            if save_per_epoch:
                saved_weights_path = saved_weights_path.join("_e" + str(i))
            weight_file = Path(saved_weights_path)
            weight_file.touch(exist_ok=True)
            model.save_weights(saved_weights_path)
            print(f"epoch {i} weights saved at {saved_weights_path}")
            weight_file = open(saved_weights_path)


    if not test_only:
        print("Trained the model!")

    if eval_train:
        train_loss = []
        batch_maps = []
        match_mses = []
        for j, data in enumerate(train_data_gen):
            print(f"Test num batch {j}")
            img_batch, y_batch = data
            yhat = model(img_batch, training=False)
            yhat = tf.reshape(yhat, [-1, grid_dim, grid_dim, hp.num_anchors, 5])
            map_batch, mse_batch = map_and_mse(y_batch, yhat, anchors, dims, threshold = 0.5)
            batch_maps.append(map_batch)
            batch_mses.append(mse_batch)
        print(f"Testing loss {np.mean(test_loss)}")
        #Print the accuracy for testing set
        print("MAP for testing set is " + str(np.mean(batch_maps)))
        print("Quantization MSE for testing set is " + str(np.mean(batch_mses)))

    # test the model
    print("Now testing...")
    y_test = encode_all_bboxes(test_labels, anchors, dims)
    test_data_gen = img_y_to_batch(test_images, y_test, hp.batch_size)
    test_loss = []
    batch_maps = []
    batch_mses = []
    
    for j, data in enumerate(test_data_gen):
        print(f"Test num batch {j}")
        img_batch, y_batch = data
        yhat = model(img_batch, training=False)
        yhat = tf.reshape(yhat, [-1, grid_dim, grid_dim, hp.num_anchors, 5])
        loss = yolo_loss(y_batch, yhat, hp.lambda_coord, hp.lambda_noobj, anchors, dims)
        test_loss.append(loss)
        map_batch, mse_batch = map_and_mse(y_batch, yhat, anchors, dims, threshold = 0.5)
        batch_maps.append(map_batch)
        batch_mses.append(mse_batch)
    print(f"Testing loss {np.mean(test_loss)}")
    #Print the accuracy for testing set
    print("MAP for testing set is " + str(np.mean(batch_maps)))
    print("Quantization MSE for testing set is " + str(np.mean(batch_mses)))


if __name__ == "__main__":

    main(saved_weights_path="saved_weights/full_50ep_1lc_1ln_0.5th",
         save_per_epoch=True,
         retrain=False,
         eval_train=True,
         test_only=False)
