import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
from importlib import reload
from utils import *

class LaneDetector():
    def __init__(self, calibration_dir = "camera_cal", test_imgs_dir = "test_images", output_imgs_dir = "output_images", output_videos_dir = "output_videos", cx=9, cy=6):
        self.calibration_dir = calibration_dir
        self.test_imgs_dir = test_imgs_dir
        self.output_imgs_dir = output_imgs_dir
        self.output_videos_dir = output_videos_dir
        
        self.cal_imgs_paths = glob.glob(calibration_dir + "/*.jpg")
        self.test_imgs_paths = glob.glob(test_imgs_dir + "/*.jpg")

        self.cx = cx
        self.cy = cy
        self.obj_pts, self.img_pts = self.find_image_object_points()

        self.test_imgs_names = [img_path.split("/")[-1].split(".")[0] for img_path in self.test_imgs_paths]
        self.undist_test_imgs_names = [f'undistorted_{img_name}' for img_name in self.test_imgs_names]
        self.test_imgs = [load_image(img_path) for img_path in self.test_imgs_paths]
        self.undist_test_imgs = [self.undistort_imge(img) for img in self.test_imgs]

        
        
    # show corners in calibration image
    def show_corners_on_chess_board(self, n=2):
        # show a sample image of calibration board
        cal_img_path = self.cal_imgs_paths[n]
        cal_img = load_image(cal_img_path)
        gray_image = to_grayscale(cal_img)
        
        # find corners
        ret, corners = cv2.findChessboardCorners(gray_image, (self.cx, self.cy), None)

        # draw corners on image
        c_img = cv2.drawChessboardCorners(cal_img, (self.cx, self.cy), corners, ret)
        plt.axis('off')
        plt.imshow(c_img)
        plt.show()
        return
    
    def find_image_object_points(self):
        obj_pts = []
        img_pts = []

        obj_pt = np.zeros((self.cx * self.cy, 3), np.float32)
        obj_pt[:, :2] = np.mgrid[0:self.cx, 0:self.cy].T.reshape(-1, 2)

        for img_path in self.cal_imgs_paths:
            img = load_image(img_path)
            gray_image = to_grayscale(img)
            ret, corners = cv2.findChessboardCorners(gray_image, (self.cx, self.cy), None)

            if ret:
                img_pts.append(corners)
                obj_pts.append(obj_pt)
        
        return obj_pts, img_pts

    def undistort_imge(self, img):
        gray_image = to_grayscale(img).shape[::-1]
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.obj_pts, self.img_pts, gray_image, None, None)
        undist = cv2.undistort(img, mtx, dist, None, mtx)
        return undist

    def threshold_img(self, img, channel, threshold=(0, 255)):
        img_channel = img[:, :, channel]
        if threshold is None:
            return img_channel
        
        mask = np.zeros_like(img_channel)
        mask[ (threshold[0] <= img_channel) and (threshold[1] >= img_channel) ] = 1
        return mask