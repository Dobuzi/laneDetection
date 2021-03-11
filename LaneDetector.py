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

        left_line, right_line = self.compute_lanes(masked_img_perspective)
        left_radius, right_radius, center_offset = self.compute_lane_curvature(left_line, right_line)

        print(left_line.polynomial_coeff, right_line.polynomial_coeff)

    def compute_lanes(self, warped_img, threshold=.85):
        hist = np.sum(warped_img[warped_img.shape[0]//2:,:], axis=0)
        mid_point = np.int(hist.shape[0]//2)
        left_point = np.argmax(hist[:mid_point])
        right_point = mid_point + np.argmax(hist[mid_point:])

        window_height = np.int(warped_img.shape[0]//self.sliding_windows_per_line)
        
        nonzero = warped_img.nonzero()
        nonzero_x, nonzero_y = np.array(nonzero[1]), np.array(nonzero[0])
        nonzero_rate = .0
        
        margin = self.sliding_window_half_width
        min_px = self.sliding_window_recenter_threshold

        left_line_indices, right_line_indices = [], []

        left_line, right_line = Line(), Line()

        if self.prev_left_line is not None and self.prev_right_line is not None:
            left_quadratic = make_quadratic(self.prev_left_line.polynomial_coeff, nonzero_y)
            right_quadratic = make_quadratic(self.prev_right_line.polynomial_coeff, nonzero_y)

            left_line_indices = (nonzero_x > (left_quadratic - margin)) & (nonzero_x < (left_quadratic + margin))
            right_line_indices = (nonzero_x > (right_quadratic - margin)) & (nonzero_x < (right_quadratic + margin))

            nonzero_rate = (len(left_line_indices) + len(right_line_indices)) / len(nonzero_y)
            print(f'[Prev_lane] Found rate: {nonzero_rate}')

        if nonzero_rate < threshold:
            print(f"Non-zeros ({nonzero_rate}) are under threshold. Let's slide window.")
            left_line_indices, right_line_indices = [], []
            
            for i in range(self.sliding_windows_per_line):
                window_y = (warped_img.shape[0] - (i+1) * window_height, warped_img.shape[0] - i * window_height)
                window_x_left = (left_point - margin, left_point + margin)
                window_x_right = (right_point - margin, right_point + margin)

                left_line.windows.append([(window_x_left[0], window_y[0]), (window_x_left[1], window_y[1])])
                right_line.windows.append([(window_x_right[0], window_y[0]), (window_x_right[1], window_y[1])])

                left_line_idx = ((nonzero_y >= window_y[0]) & (nonzero_y < window_y[1]) & \
                                 (nonzero_x >= window_x_left[0]) & (nonzero_x < window_x_left[1])).nonzero()[0]
                right_line_idx = ((nonzero_y >= window_y[0]) & (nonzero_y < window_y[1]) & \
                                 (nonzero_x >= window_x_right[0]) & (nonzero_x < window_x_right[1])).nonzero()[0]
                
                left_line_indices.append(left_line_idx)
                right_line_indices.append(right_line_idx)

                if len(left_line_idx) > min_px:
                    left_point = np.int(np.mean(nonzero_x[left_line_idx]))
                if len(right_line_idx) > min_px:
                    right_point = np.int(np.mean(nonzero_x[right_line_idx]))
            
            left_line_indices = np.concatenate(left_line_indices)
            right_line_indices = np.concatenate(right_line_indices)

            nonzero_rate = (len(left_line_indices) + len(right_line_indices)) / len(nonzero_y)
            print(f'[Sliding windows] Found rate: {nonzero_rate}')
        
        left = (nonzero_x[left_line_indices], nonzero_y[left_line_indices])
        right = (nonzero_x[right_line_indices], nonzero_y[right_line_indices])
        
        left_coeff = np.polyfit(left[1], left[0], 2)
        right_coeff = np.polyfit(right[1], right[0], 2)

        left_line.polynomial_coeff = left_coeff
        right_line.polynomial_coeff = right_coeff

        if not self.prev_left_lines.append(left_line):
            left_coeff = self.prev_left_lines.get_smoothed_polynomial()
            left_line.polynomial_coeff = left_coeff
            self.prev_left_lines.append(left_line, force=True)
            print(f'revised left poly line {left_coeff}')
        
        if not self.prev_right_lines.append(right_line):
            right_coeff = self.prev_right_lines.get_smoothed_polynomial()
            right_line.polynomial_coeff = right_coeff
            self.prev_right_lines.append(right_line, force=True)
            print(f'revised right poly line {right_coeff}')
        
        plot_y = np.linspace(0, warped_img.shape[0]-1, warped_img.shape[0])
        left_fit_x = make_quadratic(left_coeff, plot_y)
        right_fit_x = make_quadratic(right_coeff, plot_y)

        left_line.line_fit_x = left_fit_x
        left_line.non_zero_x = left[0]
        left_line.non_zero_y = left[1]

        right_line.line_fit_x = right_fit_x
        right_line.non_zero_x = right[0]
        right_line.non_zero_y = right[1]

        return (left_line, right_line)

    def compute_lane_curvature(self, left_line, right_line):
        plot_y = self.plot_y
        y_eval = np.max(plot_y)

        left_x = left_line.line_fit_x
        right_x = right_line.line_fit_x

        left_scaled = np.polyfit(plot_y * self.y_m_per_px, left_x * self.x_m_per_px, 2)
        right_scaled = np.polyfit(plot_y * self.y_m_per_px, right_x * self.x_m_per_px, 2)

        left_radius = ((1 + (2 * left_scaled[0] * y_eval * self.y_m_per_px + left_scaled[1])**2)**1.5) / np.abs(2 * left_scaled[0])
        right_radius = ((1 + (2 * right_scaled[0] * y_eval * self.y_m_per_px + right_scaled[1])**2)**1.5) / np.abs(2 * right_scaled[0])

        left_coeff = left_line.polynomial_coeff
        right_coeff = right_line.polynomial_coeff

        center_offset_img_space = (make_quadratic(left_coeff, y_eval) + make_quadratic(right_coeff, y_eval)) / 2 - self.lane_center_px_perspective
        center_offset_real_world_m = center_offset_img_space * self.x_m_per_px

        return left_radius, right_radius, center_offset_real_world_m