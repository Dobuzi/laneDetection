from utils import *
from LaneDetector import LaneDetector

if __name__ == '__main__':
    '''
    Pipe line
    1. Make the instance of LaneDetector
    '''
    laneDetector = LaneDetector()
    # show_image_list(
    #     list(zip(laneDetector.test_imgs, laneDetector.undist_test_imgs)),
    #     list(zip(laneDetector.test_imgs_names, laneDetector.undist_test_imgs_names)),
    #     "Undistortion test",
    #     fig_size=(12, 20), 
    #     show_ticks=False)

    # show_image_list(
    #     laneDetector.color_spaces,
    #     laneDetector.color_spaces_labels,
    #     "Color Channels: RGB - HLS - HSV - LAB",
    #     cols=3,
    #     fig_size=(15, 10),
    #     show_ticks=False)

    compare_2_images(
        [laneDetector.undist_test_img, laneDetector.hls_white_yellow_img],
        ["Undistorted Image", "HLS White Yellow Masked Image"])