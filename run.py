import hyperparameters as hp
import numpy as np
import time
import tensorflow as tf
from pathlib import Path

from anchors import *
from models import run_yolov4, run_one_layer
from preprocess import get_data
from train import *
from visualization import plot_boxes

def img_y_to_batch(images_dict, y, batch_size=32):
    """
    Convert images from dictionaries to
    a tensor of (batch size, img height, img width, number of color channels)
    """
    img_list = np.asarray(list(images_dict.values()))
    shuffled_indices = np.arange(len(y))
    np.random.shuffle(shuffled_indices)
    y = np.array(y)[shuffled_indices]
    img_list = img_list[shuffled_indices]

    print(f"num of images {len(img_list)}")
    last_i = 0
    for i in range(0, len(img_list), batch_size):
        if i + batch_size > len(img_list):
            continue
        batch_imgs = img_list[i:i+batch_size]
        yield batch_imgs, y[i:i+batch_size]
        last_i += batch_size
    #
    # if last_i < len(img_list):
    #     yield np.asarray(img_list[last_i:]), y[last_i:]
    #

def main(saved_weights_path="saved_weights/new_experiment",
         save_per_epoch=False,
         retrain=True,
         eval_train=True,
         test_only=False,
         visualize=False):
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
    # output = run_one_layer(input)

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

    # TODO: debugged till here! Likely in encode/decode
    y = encode_all_bboxes(train_labels, anchors, dims)


    start_time = time.time()
    train_data_gen = img_y_to_batch(train_images, y, hp.batch_size)
    end_time = time.time()
    print(f'Image loading took {end_time - start_time}s.')

    optimizer = tf.keras.optimizers.Adam()

    # train the model
    if not test_only:
        # model.summary()
        print("Now training the model...")
    start_time = time.time()

    for i in range(num_epochs):
        loss = []
        train_batch_maps = []
        train_batch_mses = []
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

                # If it's the last epoch and we want to evaluate how our training has done
                if eval_train and i == num_epochs - 1:
                    # TODO: debug
                    map_batch, mse_batch = map_and_mse(y_batch, yhat, anchors, dims, threshold=0.5)
                    train_batch_maps.append(map_batch)
                    train_batch_mses.append(mse_batch)

                # if j % 5 == 0 and visualize and i == num_epochs - 1:
                if visualize and i == num_epochs - 1:
                    # Visualize a few images, and print the boxes
                    true_yxyx, pred_yxyx = box_yxyx(y, yhat, anchors, dims, threshold=0.75, iou=0.7)
                    for img_num in [0]:
                        if len(pred_yxyx[img_num]) == 0:
                            print("No predicted boxes")
                        else:
                            plot_boxes(img_batch[img_num], true_yxyx[img_num], pred_yxyx[img_num])

                # Print out a couple of boxes every 5 batches
                if j % 5 == 0:
                    true_yxyx, pred_yxyx = box_yxyx(y[:2], yhat[:2], anchors, dims, threshold=0.75, iou=0.7)
                    true_yxyx, pred_yxyx = true_yxyx[0].numpy().tolist(), pred_yxyx[0].numpy().tolist()

                    print("True boxes:", true_yxyx)
                    if len(pred_yxyx) >= 5:
                        print(f"Predicted boxes with confidence >= 0.75 first 5 of {len(pred_yxyx)}:", pred_yxyx[:5])
                    else:
                        print("Predicted boxes with confidence >= 0.75:", pred_yxyx)

                # print('WE GOT A TRAINING STEP IN PEOPLE')
            loss.append(curr_loss)
        print(f"epoch = {i} loss = {np.mean(loss)}")
        curr_time = time.time()
        print(f"epoch {i} took {curr_time - start_time}s")
        # If it's the last epoch and we want to evaluate how our training has done
        if eval_train and i == num_epochs - 1:
            print("MAP for training set is " + str(np.mean(train_batch_maps)))
            print("Quantization MSE for training set is " + str(np.mean(train_batch_mses)))

        start_time = curr_time
        #reset the data generator
        train_data_gen = img_y_to_batch(train_images, y, hp.batch_size)

        # Save weights either per epoch or if last epoch
        if save_per_epoch or i == num_epochs - 1:
            to_save_at = saved_weights_path
            if save_per_epoch:
                to_save_at = saved_weights_path + "_e" + str(i)

            print("SAVING WEIGHTS AT " + to_save_at)
            weight_file = Path(to_save_at)
            weight_file.touch(exist_ok=True)
            model.save_weights(weight_file)
            print(f"epoch {i} weights saved at {to_save_at}")

    if not test_only:
        print("Trained the model!")

    # test the model
    print("Now testing...")
    y_test = encode_all_bboxes(test_labels, anchors, dims)
    test_data_gen = img_y_to_batch(test_images, y_test, hp.batch_size)
    test_loss = []
    test_batch_maps = []
    test_batch_mses = []
    
    for j, data in enumerate(test_data_gen):
        print(f"Test num batch {j}")
        img_batch, y_batch = data
        yhat = model(img_batch, training=False)
        yhat = tf.reshape(yhat, [-1, grid_dim, grid_dim, hp.num_anchors, 5])
        loss = yolo_loss(y_batch, yhat, hp.lambda_coord, hp.lambda_noobj, anchors, dims)
        test_loss.append(loss)

        map_batch, mse_batch = map_and_mse(y_batch, yhat, anchors, dims, threshold = 0.5)
        test_batch_maps.append(map_batch)
        test_batch_mses.append(mse_batch)

        # Visualize a few images, and print the boxes
        if j in [0, 2]:
            true_yxyx, pred_yxyx = box_yxyx(y, yhat, anchors, dims, threshold=0.9999, iou=0.7)
            for img_num in [0]:
                if len(pred_yxyx) == 0:
                    print("No predicted boxes")
                else:
                    plot_boxes(img_batch[img_num], true_yxyx[img_num], pred_yxyx[img_num])

    print(f"Testing loss {np.mean(test_loss)}")
    #Print the accuracy for testing set
    print("MAP for testing set is " + str(np.mean(test_batch_maps)))
    print("Quantization MSE for testing set is " + str(np.mean(test_batch_mses)))


if __name__ == "__main__":
    main(saved_weights_path="saved_weights/full_50ep_1lc_1ln_0.5th",
         save_per_epoch=True,
         retrain=True,
         eval_train=False,
         test_only=False,
         visualize=False # Visualize for the last epoch, every 5 batches
         )

    # main(saved_weights_path="saved_weights/onelayer",
    #      save_per_epoch=False,
    #      retrain=True,
    #      eval_train=False,
    #      test_only=False,
    #      visualize=False)
