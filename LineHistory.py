import numpy as np
from collections import deque
from utils import *

class LineHistory:
    def __init__(self, queue_depth=2, test_points=[50, 300, 500, 700], poly_max_deviation_distance=150):
        self.lane_lines = deque(maxlen=queue_depth)
        self.smoothed_poly = None
        self.test_points = test_points
        self.poly_max_deviation_distance = poly_max_deviation_distance

    def get_smoothed_polynomial(self):
        all_coeffs = np.asarray([lane_line.polynomial_coeff for lane_line in self.lane_lines])
        return np.mean(all_coeffs, axis=0)

    def append(self, lane_line, force=False):
        if force or len(self.lane_lines) == 0:
            self.lane_lines.append(lane_line)
            self.smoothed_poly = self.get_smoothed_polynomial()
            return True
        
        test_y_smooth = np.asarray([make_quadratic(self.smoothed_poly, pt) for pt in self.test_points])
        test_y_new = np.asarray([make_quadratic(lane_line.polynomial_coeff, pt) for pt in self.test_points])

        dist = np.abs(test_y_smooth - test_y_new)

        max_dist = dist[np.argmax(dist)]
        
        if max_dist > self.poly_max_deviation_distance:
            print('*** OVER MAX DISTANCE ***')
            print(f'distance {max_dist} > max_distance {self.poly_max_deviation_distance}')
            return False
        
        self.lane_lines.append(lane_line)
        self.smoothed_poly = self.get_smoothed_polynomial()
        return True