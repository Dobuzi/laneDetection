import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt

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

def show_image_list(img_list, img_labels, title, cols=2, fig_size=(15, 15) ,show_ticks=True):
    rows = len(img_list)
    cmap = None
    
    fig, axes = plt.subplots(rows, cols, fig_size=fig_size)

    for i in range(0, rows):
        for j in range(0, cols):
            ax = axes[i, j] if rows > 1 else axes[j]
            
            img_name = img_labels[i][j]
            img = img_list[i][j]

            if len(img.shape) < 3 or img.shape[-1] < 3:
                cmap = "gray"
                img = np.reshape(img, (img.shape[0], img.shape[1]))
            
            if not show_ticks:
                ax.axis("off")
            
            ax.imshow(img, cmap=cmap)
            ax.set_title(img_name)
    
    fig.subtitle(title, fontsize=12, fontweight='bold', y=1)
    fig.tight_layout()
    plt.show()

    return