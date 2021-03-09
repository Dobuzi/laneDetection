from utils import *
from LaneDetector import LaneDetector
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    '''
    Pipe line
    1. Make the instance of LaneDetector
    2. Check Unditortion Image for test
    3. Check Color spaces
    4. Check White and Yellow mask
    5. Check Sobel filter parameters
    6. Apply combined filters
    '''
    # 1. Make the instance of LaneDetector
    laneDetector = LaneDetector(n_sample=7)

    # 2. Check Unditortion Image for test
    # show_image_list(
    #     list(zip(laneDetector.test_imgs, laneDetector.undist_test_imgs)),
    #     list(zip(laneDetector.test_imgs_names, laneDetector.undist_test_imgs_names)),
    #     "Undistortion test",
    #     fig_size=(12, 20), 
    #     show_ticks=False)

    # 3. Check Color spaces
    # show_image_list(
    #     laneDetector.color_spaces,
    #     laneDetector.color_spaces_labels,
    #     "Color Channels: RGB - HLS - HSV - LAB",
    #     cols=3,
    #     fig_size=(15, 10),
    #     show_ticks=False)

    # 4. Check White and Yellow mask
    # compare_2_images(
    #     [laneDetector.undist_test_img, laneDetector.hls_white_yellow_img],
    #     ["Undistorted Image", "HLS White Yellow Masked Image"])

    # 5. Check Sobel filter parameters

    # sobel_x_imgs, sobel_x_labels = laneDetector.test_sobel_filter(laneDetector.abs_sobel)
    # show_image_list(sobel_x_imgs, sobel_x_labels, "Sobel X-dir masks", cols=3, show_ticks=False)
    
    # sobel_y_imgs, sobel_y_labels = laneDetector.test_sobel_filter(laneDetector.abs_sobel, x_dir=False)
    # show_image_list(sobel_y_imgs, sobel_y_labels, "Sobel Y-dir masks", cols=3, fig_size=(10, 10), show_ticks=False)

    # sobel_xy_imgs, sobel_xy_labels = laneDetector.test_sobel_filter(laneDetector.mag_sobel, x_dir=None)
    # show_image_list(sobel_xy_imgs, sobel_xy_labels, "Sobel XY magnitude masks", cols=3, fig_size=(10, 10), show_ticks=False)

    # sobel_combined_imgs, sobel_combined_labels = laneDetector.test_sobel_filter(laneDetector.combine_sobel, x_dir=None, angle=True)
    # show_image_list(sobel_combined_imgs, sobel_combined_labels, "Sobel Combined masks", cols=3, fig_size=(10, 10), show_ticks=False)
    
    # show_image_list(laneDetector.bins, laneDetector.bins_labels, "Color And Binary Combined Gradient And HLS (S) Theshold", fig_size=(10, 10), show_ticks=False)

    # plt.imshow(laneDetector.masked_roi_test_img)
    # plt.axis('off')
    # plt.show()

    # imgs = np.asarray(list(zip(laneDetector.undist_test_imgs, laneDetector.perspective_test_imgs)))
    # labels = np.asarray(list(zip(laneDetector.undist_test_imgs_names, laneDetector.undist_test_imgs_names)))

    # show_image_list(imgs, labels, "Undistorted and Birds View Image", fig_size=(15, 15))

    # 6. Apply combined filters
    combined_bin_imgs = np.asarray([laneDetector.get_combined_binary_masked_img(img) for img in laneDetector.undist_test_imgs])
    perspective_transform_imgs = np.asarray([laneDetector.perspective_transform(img, laneDetector.warp_src, laneDetector.warp_dst) for img in laneDetector.undist_test_imgs])
    
    solution_imgs = np.asarray([laneDetector.perspective_transform(img, laneDetector.warp_src, laneDetector.warp_dst) for img in combined_bin_imgs])

    # total_imgs = list(zip(combined_bin_imgs, perspective_transform_imgs, solution_imgs))
    # total_imgs_labels = list(zip(laneDetector.undist_test_imgs_names, laneDetector.undist_test_imgs_names, laneDetector.undist_test_imgs_names))

    # show_image_list(total_imgs, total_imgs_labels, "Combined bin and Perspective Transform Images", cols=3, fig_size=(15, 15))

    img_example = solution_imgs[1]
    hist = np.sum(img_example[img_example.shape[0]//2:, :], axis=0)
    
    fig, ax = plt.subplots(1, 2, figsize=(15, 4))
    ax[0].imshow(img_example, cmap='gray')
    ax[0].axis('off')
    ax[0].set_title('Image')
    
    ax[1].plot(hist)
    ax[1].set_title('Histogram of pixel intensity (bottom half)')

    plt.show()