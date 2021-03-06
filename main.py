from utils import *
from LaneDetector import LaneDetector
import matplotlib.pyplot as plt

if __name__ == '__main__':
    '''
    Pipe line
    1. Make the instance of LaneDetector
    2. Check Unditortion Image for test
    3. Check Color spaces
    4. Check White and Yellow mask
    5. Check Sobel filter parameters
    '''
    # 1. Make the instance of LaneDetector
    laneDetector = LaneDetector()

    # 2. Check Unditortion Image for test
    # show_image_list(
    #     list(zip(laneDetector.test_imgs, laneDetector.undist_test_imgs)),
    #     list(zip(laneDetector.test_imgs_names, laneDetector.undist_test_imgs_names)),
    #     "Undistortion test",
    #     fig_size=(12, 20), 
    #     show_ticks=False)

    # 3. Check Color spaces
    # show_image_list(
    #     laneDetector.color_spaces,
    #     laneDetector.color_spaces_labels,
    #     "Color Channels: RGB - HLS - HSV - LAB",
    #     cols=3,
    #     fig_size=(15, 10),
    #     show_ticks=False)

    # 4. Check White and Yellow mask
    # compare_2_images(
    #     [laneDetector.undist_test_img, laneDetector.hls_white_yellow_img],
    #     ["Undistorted Image", "HLS White Yellow Masked Image"])

    # 5. Check Sobel filter parameters
    # L_img = laneDetector.to_lab(laneDetector.undist_test_img)[:,:,0]

    # sobel_x_imgs, sobel_x_labels = laneDetector.test_sobel_filter(L_img)
    # show_image_list(
    #     sobel_x_imgs,
    #     sobel_x_labels,
    #     "Sobel X-dir masks",
    #     cols=3,
    #     show_ticks=False
    # )

    # sobel_y_imgs, sobel_y_labels = laneDetector.test_sobel_filter(L_img, x_dir=False)
    # show_image_list(
    #     sobel_y_imgs,
    #     sobel_y_labels,
    #     "Sobel Y-dir masks",
    #     cols=3,
    #     fig_size=(10, 10),
    #     show_ticks=False
    # )