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

        self.undist_test_img_gray = self.to_lab(self.undist_test_img)[:,:,0]
        self.sobel_x_best = self.abs_sobel(self.undist_test_img, kernel_size=15, threshold=(20, 120))
        self.sobel_y_best = self.abs_sobel(self.undist_test_img, x_dir=False, kernel_size=15, threshold=(20, 120))
        self.sobel_xy_best = self.mag_sobel(self.undist_test_img, kernel_size=15, threshold=(80, 200))
        self.sobel_combined_best = self.combine_sobel(self.undist_test_img, self.sobel_x_best, self.sobel_y_best, self.sobel_xy_best, kernel_size=15, threshold=(np.pi/4, np.pi/2))
        self.color_bin = self.make_color_bin()
        self.combined_bin = self.make_combined_bin
        self.bins = [[self.color_bin, self.combined_bin]]
        self.bins_labels = np.asarray([["Stacked Thresholds", "Combined Color And Gradient Masks"]])


        # define the region of interest (ROI)
        bottom_px, right_px = (self.undist_test_img.shape[0] - 1, self.undist_test_img.shape[1] - 1)

        size = {'bottom': {'left': 200, 'right': 1100},
                'top': {'left': 610, 'right': 680},
                'height': 280}
        gap = 100

        self.roi = np.array([
            [size['bottom']['left'], bottom_px], 
            [size['top']['left'], bottom_px - size['height']], 
            [size['top']['right'], bottom_px - size['height']],
            [size['bottom']['right'], bottom_px]], np.int32)
        
        self.warp_src = self.roi.astype(np.float32)
        
        self.warp_dst = np.array([
            [size['bottom']['left'], bottom_px],
            [size['bottom']['left'], 0],
            [size['bottom']['right'] - gap, 0],
            [size['bottom']['right'] - gap, bottom_px]], np.float32)

        self.masked_roi_test_img = self.mask_roi(self.undist_test_img)
        self.perspective_test_img = self.perspective_transform(self.undist_test_img, self.warp_src, self.warp_dst)
        self.perspective_test_imgs = [self.perspective_transform(img, self.warp_src, self.warp_dst) for img in self.undist_test_imgs]

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
    
    def abs_sobel(self, img_gray, x_dir=True, kernel_size=3, threshold=(0, 255)):
        sobel = cv2.Sobel(img_gray, cv2.CV_64F, int(x_dir), int(not x_dir), ksize=kernel_size)
        
        sobel_abs = np.absolute(sobel)
        sobel_scaled = np.uint8(255 * sobel / np.max(sobel_abs))

        gradient_mask = np.zeros_like(sobel_scaled)
        gradient_mask[(threshold[0] <= sobel_scaled) & (sobel_scaled <= threshold[1])] = 1

        return gradient_mask
    
    def mag_sobel(self, img_gray, kernel_size=3, threshold=(0, 255)):
        sobel_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=kernel_size)
        sobel_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=kernel_size)

        sobel_xy = np.sqrt(np.square(sobel_x) + np.square(sobel_y))
        scaled_sobel_xy = np.uint8(255 * sobel_xy / np.max(sobel_xy))

        sobel_xy_bin = np.zeros_like(scaled_sobel_xy)
        sobel_xy_bin[(scaled_sobel_xy >= threshold[0]) & (scaled_sobel_xy <= threshold[1])] = 1

        return sobel_xy_bin

    def dir_sobel(self, img_gray, kernel_size=3, threshold=(0, np.pi/2)):
        sobel_x_abs = np.absolute(cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=kernel_size))
        sobel_y_abs = np.absolute(cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=kernel_size))

        dir_sobel_xy = np.arctan2(sobel_x_abs, sobel_y_abs)

        bin_output = np.zeros_like(dir_sobel_xy)
        bin_output[(dir_sobel_xy >= threshold[0]) & (dir_sobel_xy <= threshold[1])] = 1

        return bin_output
    
    def combine_sobel(self, img_gray, sobel_x, sobel_y, sobel_xy, kernel_size=3, threshold=(0, np.pi/2)):
        sobel_xy_dir = self.dir_sobel(img_gray, kernel_size=kernel_size, threshold=threshold)
        combined = np.zeros_like(sobel_xy_dir)

        combined[(sobel_x == 1) | (sobel_y == 1) & (sobel_xy == 1) & (sobel_xy_dir == 1)] = 1

        return combined

    def test_sobel_filter(self, callback, x_dir=True, angle=False):
        kernel_sizes = [3, 7, 11, 15]
        thresholds = [(0, np.pi/4), (np.pi/4, np.pi/2), (np.pi/3, np.pi/2)] if angle else [(20, 120), (50, 150), (80, 200)]
             
        sobel_imgs = []
        sobel_labels = []
        for k in kernel_sizes:
            sobel_img = []
            sobel_label = []
            for th in thresholds:
                if x_dir is None:
                    sobel_img.append(callback(kernel_size=k, threshold=th))
                else:
                    sobel_img.append(callback(x_dir=x_dir, kernel_size=k, threshold=th))
                sobel_label.append(f'{k}x{k} - Threshold {th}')
            sobel_imgs.append(sobel_img)
            sobel_labels.append(sobel_label)
        
        return np.asarray(sobel_imgs), np.asarray(sobel_labels)

    def make_color_bin(self):
        color_bin = np.dstack((np.zeros_like(self.sobel_combined_best), self.sobel_combined_best, self.hls_white_yellow_img)) * 255
        color_bin = color_bin.astype(np.uint8)

        return color_bin

    def make_combined_bin(self):
        combined_bin = np.zeros_like(self.hls_white_yellow_img)
        combined_bin[(self.sobel_combined_best == 1) | (self.hls_white_yellow_img == 1)] = 1

        return combined_bin

    def mask_roi(self, img):
        mask_roi = np.copy(img)
        
        cv2.polylines(mask_roi, [self.roi], True, (0, 255, 0), 10)

        return mask_roi
    
    def compute_perspective_transform_matrices(self, src, dst):
        M = cv2.getPerspectiveTransform(src, dst)
        M_inv = cv2. getPerspectiveTransform(dst, src)

        return (M, M_inv)
    
    def perspective_transform(self, img, src, dst):
        M = cv2.getPerspectiveTransform(src, dst)
        img_size = (img.shape[1], img.shape[0])
        warped_img = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
        return warped_img

    def get_combined_binary_masked_img(self, img):
        img_gray = self.to_lab(img)[:, :, 0]
        sobel_x = self.abs_sobel(img_gray, kernel_size=15, threshold=(20, 120))
        sobel_y = self.abs_sobel(img_gray, x_dir=False, kernel_size=15, threshold=(20, 120))
        sobel_xy = self.mag_sobel(img_gray, kernel_size=15, threshold=(80, 200))
        
        sobel_combined_dir = self.combine_sobel(img_gray, sobel_x, sobel_y, sobel_xy, kernel_size=15, threshold=(np.pi/4, np.pi/2))

        hls_w_y = self.compute_hls_white_yellow_binary(img)

        combined_bin = np.zeros_like(hls_w_y)
        combined_bin[(sobel_combined_dir == 1) | (hls_w_y == 1)] = 1

        return combined_bin