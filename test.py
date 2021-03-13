import glob
from preprocess import *
from LaneDetector import *
import cv2

def calibrate(n_sample=0):
    obj_pts, img_pts = find_image_object_points()

    test_imgs_dir = "test_images"
    test_imgs_paths = glob.glob(test_imgs_dir + '/*.jpg')
    test_imgs = [load_image(img_path) for img_path in test_imgs_paths]
    undist_test_imgs = [undistort_image(img, obj_pts, img_pts) for img in test_imgs]

    test_img = test_imgs[n_sample]
    undist_test_img = undist_test_imgs[n_sample]

    warp_src, warp_dst = get_warp_src_dst(undist_test_img)

    return obj_pts, img_pts, warp_src, warp_dst

def test_image(n_sample=0):
    obj_pts, img_pts, warp_src, warp_dst = calibrate(n_sample)
    laneDetector = LaneDetector(obj_pts, img_pts, warp_src, warp_dst, 20, 100, 50)
    out_img = laneDetector.process_image(test_img)
    return out_img

def test_video(n_sample=0):
    obj_pts, img_pts, warp_src, warp_dst = calibrate()

    input_dir = 'input_videos/'
    output_dir = 'output_videos/'
    
    video_names = ['project_video_sample.mp4',
                   'project_video.mp4',
                   'harder_challenge_video.mp4',
                   'challenge_video.mp4',]
    
    input_paths = [input_dir + name for name in video_names]
    output_paths = [output_dir + name for name in video_names]

    laneDetector = LaneDetector(obj_pts, img_pts, warp_src, warp_dst, 20, 100, 10)

    cap = cv2.VideoCapture(input_paths[n_sample])
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter(output_paths[n_sample], fourcc, 20.0, (1280, 720))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Can't receiving frame (stream end?). Exiting...")
            break
        frame = laneDetector.process_image(frame)
        out.write(frame)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == ord('q'):
            break
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    return