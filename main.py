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

    # sobel_x_imgs, sobel_x_labels = laneDetector.test_sobel_filter(laneDetector.abs_sobel)
    # show_image_list(sobel_x_imgs, sobel_x_labels, "Sobel X-dir masks", cols=3, show_ticks=False)
    
    # sobel_y_imgs, sobel_y_labels = laneDetector.test_sobel_filter(laneDetector.abs_sobel, x_dir=False)
    # show_image_list(sobel_y_imgs, sobel_y_labels, "Sobel Y-dir masks", cols=3, fig_size=(10, 10), show_ticks=False)

    # sobel_xy_imgs, sobel_xy_labels = laneDetector.test_sobel_filter(laneDetector.mag_sobel, x_dir=None)
    # show_image_list(sobel_xy_imgs, sobel_xy_labels, "Sobel XY magnitude masks", cols=3, fig_size=(10, 10), show_ticks=False)

    # sobel_combined_imgs, sobel_combined_labels = laneDetector.test_sobel_filter(laneDetector.combine_sobel, x_dir=None, angle=True)
    # show_image_list(sobel_combined_imgs, sobel_combined_labels, "Sobel Combined masks", cols=3, fig_size=(10, 10), show_ticks=False)
    
    # combined_bin, combined_bin_labels = laneDetector.combine_bin()
    # show_image_list(combined_bin, combined_bin_labels, "Color And Binary Combined Gradient And HLS (S) Theshold", fig_size=(10, 10), show_ticks=False)

    plt.axis('off')
    plt.imshow(laneDetector.perspective_test_img)
    plt.show()