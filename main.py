from utils import *
from LaneDetector import LaneDetector

if __name__ == '__main__':
    '''
    Pipe line
    1. Make the instance of LaneDetector
    '''
    laneDetector = LaneDetector()
    show_image_list(
        list(zip(laneDetector.test_imgs, laneDetector.undist_test_imgs)),
        list(zip(laneDetector.test_imgs_names, laneDetector.undist_test_imgs_names)),
        "Undistortion test",
        fig_size=(12, 20), 
        show_ticks=False)