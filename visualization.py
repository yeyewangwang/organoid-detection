import matplotlib.pyplot as plt
import cv2
import numpy as np

def plot_boxes(img, true_yxyxs, pred_yxyxs):
    """
    Plot a pair of an image and a list of bounding box yxyx's.
    """
    # print("Neg vals")
    # print(img.shape, img[0][:2])
    # # Convert to 0 to 1 range
    img = img.astype(float) / 255
    # print(np.sum(img[img<0]))


    for yxyx in true_yxyxs:
        yxyx = yxyx.numpy()
        # print((yxyx[1], yxyx[0]), (yxyx[3], yxyx[2]))
        # Blue
        cv2.rectangle(img, (yxyx[1], yxyx[0]), (yxyx[3], yxyx[2]),  (0,0,255))
    print("Predicted boxes")
    for yxyx in pred_yxyxs:
        yxyx = yxyx.numpy().astype(int)
        print((yxyx[1], yxyx[0]), (yxyx[3], yxyx[2]))
        # Orange
        cv2.rectangle(img, (yxyx[1], yxyx[0]), (yxyx[3], yxyx[2]), (255, 128, 0), 2)
    # TODO: visualize images

    plt.imshow(img)
    plt.show()
    plt.pause(2)
    plt.close()


