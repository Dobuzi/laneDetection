import glob
from preprocess import *
from LaneDetector import *

if __name__ == '__main__':
    obj_pts, img_pts = find_image_object_points()

    n_sample = 6
    test_imgs_dir = "test_images"
    test_imgs_paths = glob.glob(test_imgs_dir + '/*.jpg')
    test_imgs = [load_image(img_path) for img_path in test_imgs_paths]
    undist_test_imgs = [undistort_image(img, obj_pts, img_pts) for img in test_imgs]
    test_img = test_imgs[n_sample]
    undist_test_img = undist_test_imgs[n_sample]

    
    warp_src, warp_dst = get_warp_src_dst(undist_test_img)

    laneDetector = LaneDetector(obj_pts, img_pts, warp_src, warp_dst, 20, 100, 50)

    laneDetector.process_image(test_img)