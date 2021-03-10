import cv2
from Line import *
from LineHistory import *
import preprocess
from utils import *

class LaneDetector():
    def __init__(self, obj_pts, img_pts,
                 perspective_src, perspective_dst,
                 sliding_windows_per_line, sliding_window_half_width, sliding_window_recenter_threshold,
                 small_img_size=(256, 144), small_img_x_offset=20, small_img_y_offset=10,
                 img_dimensions=(720, 1280), lane_width_px=800, lane_center_px_perspective=600,
                 real_world_lane_size_meters=(32, 3.7)):
        self.obj_pts = obj_pts
        self.img_pts = img_pts
        self.M = cv2.getPerspectiveTransform(perspective_src, perspective_dst)
        self.M_inv = cv2.getPerspectiveTransform(perspective_dst, perspective_src)

        self.sliding_windows_per_line = sliding_windows_per_line
        self.sliding_window_half_width = sliding_window_half_width
        self.sliding_window_recenter_threshold = sliding_window_recenter_threshold

        self.small_img_size = small_img_size
        self.small_img_x_offset = small_img_x_offset
        self.small_img_y_offset = small_img_y_offset

        self.img_dimensions = img_dimensions
        self.lane_width_px = lane_width_px
        self.lane_center_px_perspective = lane_center_px_perspective
        self.real_world_lane_size_meters = real_world_lane_size_meters

        self.y_m_per_px = self.real_world_lane_size_meters[0] / self.img_dimensions[0]
        self.x_m_per_px = self.real_world_lane_size_meters[1] / self.lane_width_px
        self.plot_y = np.linspace(0, self.img_dimensions[0] - 1, self.img_dimensions[0])

        self.prev_left_line = None
        self.prev_right_line = None

        self.prev_left_lines = LineHistory()
        self.prev_right_lines = LineHistory()

        self.total_img_count = 0

    def process_image(self, img):
        # 1. Undistort the image
        undist_img = preprocess.undistort_image(img, self.obj_pts, self.img_pts)

        #2. Produce binary image
        masked_img = preprocess.get_combined_binary_masked_img(undist_img)

        img_size = (undist_img.shape[1], undist_img.shape[0])

        undist_img_perspective = cv2.warpPerspective(undist_img, self.M, img_size, flags=cv2.INTER_LINEAR)
        masked_img_perspective = cv2.warpPerspective(masked_img, self.M, img_size, flags=cv2.INTER_LINEAR)

    def compute_lanes(self, warped_img):
        hist = np.sum(warped_img[warped_img.shape[0]//2:,:], axis=0)
        mid_point = np.int(hist.shape[0]//2)
        left_point = np.argmax(hist[:mid_point])
        right_point = mid_point + np.argmax(hist[mid_point:])

        window_height = np.int(warped_img.shape[0]//self.sliding_windows_per_line)
        
        nonzero = warped_img.nonzero()
        
        margin = self.sliding_window_half_width
        min_px = self.sliding_window_recenter_threshold

        left_line_indices, right_line_indices = [], []

        left_line, right_line = Line(), Line()

        left_coeff, right_coeff = self.prev_left_line.polynomial_coeff, self.prev_right_line.polynomial_coeff
        left_quadratic, right_quadratic = make_quadratic(left_coeff, nonzero[0]), make_quadratic(right_coeff, nonzero[0])

        if self.prev_left_line is not None and self.prev_right_line is not None:
            left_line_indices = (nonzero[1] > (left_quadratic - margin)) & (nonzero[1] < (left_quadratic + margin))
            right_line_indices = (nonzero[1] > (right_quadratic - margin)) & (nonzero[1] < (right_quadratic + margin))
        
