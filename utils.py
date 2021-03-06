import numpy as np
import matplotlib.pyplot as plt

def show_image_list(img_list, img_labels, title, cols=2, fig_size=(15, 15) ,show_ticks=True):
    rows = len(img_list)
    cmap = None
    
    fig, axes = plt.subplots(rows, cols, figsize=fig_size)

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
    
    fig.suptitle(title, fontsize=12, fontweight='bold', y=1)
    fig.tight_layout()
    plt.show()

    return

def compare_2_images(images, titles):
    fig, ax = plt.subplots(1, 2, figsize=(10, 7))
    for i, (image, title) in enumerate(zip(images, titles)):
        ax[i].imshow(image)
        ax[i].axis("off")
        ax[i].set_title(title)
    plt.show()
    return