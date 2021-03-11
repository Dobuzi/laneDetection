import glob
import cv2
import numpy as np

def to_grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def to_hsv(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

def to_hls(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

def to_lab(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

def load_image(path, to_rgb=True):
    img = cv2.imread(path)
    return img if not to_rgb else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def find_image_object_points(calibration_dir = "camera_cal", cx=9, cy=6):
    cal_imgs_paths = glob.glob(calibration_dir + "/*.jpg")

    obj_pts = []
    img_pts = []

    obj_pt = np.zeros((cx * cy, 3), np.float32)
    obj_pt[:, :2] = np.mgrid[0:cx, 0:cy].T.reshape(-1, 2)

    for img_path in cal_imgs_paths:
        img = load_image(img_path)
        gray_image = to_grayscale(img)
        ret, corners = cv2.findChessboardCorners(gray_image, (cx, cy), None)

        if ret:
            img_pts.append(corners)
            obj_pts.append(obj_pt)
    
    return obj_pts, img_pts

def undistort_image(img, obj_pts, img_pts):
    gray_image = to_grayscale(img).shape[::-1]
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_pts, img_pts, gray_image, None, None)
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist

def compute_hls_white_yellow_binary(img):
    hls_img = to_hls(img)
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

def abs_sobel(img_gray, x_dir=True, kernel_size=3, threshold=(0, 255)):
    sobel = cv2.Sobel(img_gray, cv2.CV_64F, int(x_dir), int(not x_dir), ksize=kernel_size)
    
    sobel_abs = np.absolute(sobel)
    sobel_scaled = np.uint8(255 * sobel / np.max(sobel_abs))

    gradient_mask = np.zeros_like(sobel_scaled)
    gradient_mask[(threshold[0] <= sobel_scaled) & (sobel_scaled <= threshold[1])] = 1

    return gradient_mask

def mag_sobel(img_gray, kernel_size=3, threshold=(0, 255)):
    sobel_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=kernel_size)
    sobel_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=kernel_size)

    sobel_xy = np.sqrt(np.square(sobel_x) + np.square(sobel_y))
    scaled_sobel_xy = np.uint8(255 * sobel_xy / np.max(sobel_xy))

    sobel_xy_bin = np.zeros_like(scaled_sobel_xy)
    sobel_xy_bin[(scaled_sobel_xy >= threshold[0]) & (scaled_sobel_xy <= threshold[1])] = 1

    return sobel_xy_bin

def dir_sobel(img_gray, kernel_size=3, threshold=(0, np.pi/2)):
    sobel_x_abs = np.absolute(cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=kernel_size))
    sobel_y_abs = np.absolute(cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=kernel_size))

    dir_sobel_xy = np.arctan2(sobel_x_abs, sobel_y_abs)

    bin_output = np.zeros_like(dir_sobel_xy)
    bin_output[(dir_sobel_xy >= threshold[0]) & (dir_sobel_xy <= threshold[1])] = 1

    return bin_output

def combine_sobel(img_gray, sobel_x, sobel_y, sobel_xy, kernel_size=3, threshold=(0, np.pi/2)):
    sobel_xy_dir = dir_sobel(img_gray, kernel_size=kernel_size, threshold=threshold)
    combined_sobel = np.zeros_like(sobel_xy_dir)

    combined_sobel[(sobel_x == 1) | (sobel_y == 1) & (sobel_xy == 1) & (sobel_xy_dir == 1)] = 1

    return combined_sobel

def get_combined_binary_masked_img(img):
    img_gray = to_lab(img)[:, :, 0]
    sobel_x = abs_sobel(img_gray, kernel_size=15, threshold=(20, 120))
    sobel_y = abs_sobel(img_gray, x_dir=False, kernel_size=15, threshold=(20, 120))
    sobel_xy = mag_sobel(img_gray, kernel_size=15, threshold=(80, 200))
    
    sobel_combined_dir = combine_sobel(img_gray, sobel_x, sobel_y, sobel_xy, kernel_size=15, threshold=(np.pi/4, np.pi/2))

    hls_w_y = compute_hls_white_yellow_binary(img)

    combined_bin = np.zeros_like(hls_w_y)
    combined_bin[(sobel_combined_dir == 1) | (hls_w_y == 1)] = 1

    return combined_bin

def get_warp_src_dst(undsitorted_img):
    bottom_px, right_px = (undsitorted_img.shape[0] - 1, undsitorted_img.shape[1] - 1)

    size = {'bottom': {'left': 200, 'right': 1100},
            'top': {'left': 610, 'right': 680},
            'height': 280}
    gap = 100

    roi = np.array([
        [size['bottom']['left'], bottom_px], 
        [size['top']['left'], bottom_px - size['height']], 
        [size['top']['right'], bottom_px - size['height']],
        [size['bottom']['right'], bottom_px]], np.int32)
    
    warp_src = roi.astype(np.float32)
    
    warp_dst = np.array([
        [size['bottom']['left'], bottom_px],
        [size['bottom']['left'], 0],
        [size['bottom']['right'] - gap, 0],
        [size['bottom']['right'] - gap, bottom_px]], np.float32)
    
    return warp_src, warp_dst