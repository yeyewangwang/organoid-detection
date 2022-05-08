import matplotlib.pyplot as plt
import cv2

def plot_boxes(img, true_yxyxs, pred_yxyxs):
    """
    Plot a pair of an image and a list of bounding box yxyx's.
    """
    print("True boxes")
    for yxyx in true_yxyxs:
        yxyx = yxyx.numpy()
        print((yxyx[1], yxyx[0]), (yxyx[3], yxyx[2]))
        # cv2.rectangle(img, (yxyx[1], yxyx[0]), (yxyx[3], yxyx[2]),  (100, 100, 0))
    print("Predicted boxes")
    for yxyx in pred_yxyxs:
        yxyx = yxyx.numpy()
        print((yxyx[1], yxyx[0]), (yxyx[3], yxyx[2]))

         # cv2.rectangle(img, (yxyx[1], yxyx[0]), (yxyx[3], yxyx[2]), (255, 128, 0), 2)
    # TODO: visualize images
    # plt.imshow(img)
    # plt.show()
    # plt.pause(2)
    # plt.close()


