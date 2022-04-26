import keras
import numpy as np
import tensorflow as tf
from preprocess import get_data
#from models import YOLOV4

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

    #train the model

    #test the model

if __name__ == "__main__":
    #parse_args()

    main()
