import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
from importlib import reload
from utils import *

class LaneDetector():
    def __init__(self, calibration_dir = "camera_cal", test_imgs_dir = "test_images", output_imgs_dir = "output_images", output_videos_dir = "output_videos", cx=9, cy=6, n_sample=0):
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
        self.test_imgs = [self.load_image(img_path) for img_path in self.test_imgs_paths]
        self.undist_test_imgs = [self.undistort_image(img) for img in self.test_imgs]

        self.undist_test_img = self.undist_test_imgs[n_sample]
        self.color_spaces, self.color_spaces_labels = self.make_color_spaces(self.undist_test_img)
        self.hls_white_yellow_img = self.compute_hls_white_yellow_binary(self.undist_test_img)

    # show corners in calibration image
    def show_corners_on_chess_board(self, n=2):
        # show a sample image of calibration board
        cal_img_path = self.cal_imgs_paths[n]
        cal_img = self.load_image(cal_img_path)
        gray_image = self.to_grayscale(cal_img)
        
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
            img = self.load_image(img_path)
            gray_image = self.to_grayscale(img)
            ret, corners = cv2.findChessboardCorners(gray_image, (self.cx, self.cy), None)

            if ret:
                img_pts.append(corners)
                obj_pts.append(obj_pt)
        
        return obj_pts, img_pts

    def undistort_image(self, img):
        gray_image = self.to_grayscale(img).shape[::-1]
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.obj_pts, self.img_pts, gray_image, None, None)
        undist = cv2.undistort(img, mtx, dist, None, mtx)
        return undist

    def mask_image(self, img, channel, threshold=(0, 255)):
        img_channel = img[:, :, channel]
        if threshold is None:
            return img_channel
        
        mask = np.zeros_like(img_channel)
        mask[threshold[0] <= img_channel <= threshold[1]] = 1
        return mask
    
    def to_grayscale(self, img):
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    def to_hsv(self, img):
        return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    def to_hls(self, img):
        return cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

    def to_lab(self, img):
        return cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

    def load_image(self, path, to_rgb=True):
        img = cv2.imread(path)
        return img if not to_rgb else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    def filter_3_channel(self, img):
        return [self.mask_image(img, 0, threshold=None),
                 self.mask_image(img, 1, threshold=None),
                 self.mask_image(img, 2, threshold=None)]

    def make_color_spaces(self, img):
        rgb_filter = np.asarray([self.filter_3_channel(img)])
        rgb_labels = np.asarray([["Red", "Green", "Blue"]])

        hls_img = self.to_hls(img)
        hls_filter = np.asarray([self.filter_3_channel(hls_img)])
        hls_labels = np.asarray([["Hue", "Lightness", "Saturation"]])

        hsv_img = self.to_hsv(img)
        hsv_filter = np.asarray([self.filter_3_channel(hsv_img)])
        hsv_labels = np.asarray([["Hue", "Saturation", "Value"]])

        lab_img = self.to_lab(img)
        lab_filter = np.asarray([self.filter_3_channel(lab_img)])
        lab_labels = np.asarray([["Lightness", "Green-Red (A)", "Blue-Yellow (B)"]])

        color_spaces = np.concatenate((rgb_filter, hls_filter, hsv_filter, lab_filter))
        color_spaces_labels = np.concatenate((rgb_labels, hls_labels, hsv_labels, lab_labels))

        return color_spaces, color_spaces_labels
    
    def compute_hls_white_yellow_binary(self, img):
        hls_img = self.to_hls(img)
        hue, lightness, saturation = hls_img[:,:,0], hls_img[:,:,1], hls_img[:,:,2]

        hls_img_w = np.zeros_like(hls_img[:,:,0])
        hls_img_w[(0 <= hue) & (hue <= 255) & \
                  (200 <= lightness) & (lightness <= 255) & \
                  (0 <= saturation) & (saturation <= 255)] = 1

        hls_img_y = np.zeros_like(hls_img[:,:,0])
        hls_img_y[(15 <= hue) & (hue <= 35) & \
                  (30 <= lightness) & (lightness <= 204) & \
                  (115 <= saturation) & (saturation <= 255)] = 1
        
        hls_img_w_y = np.zeros_like(hls_img[:,:,0])
        hls_img_w_y[(hls_img_w == 1) | (hls_img_y == 1)] = 1

        return hls_img_w_y