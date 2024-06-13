import os
import cv2
import numpy as np

import matplotlib.pyplot as plt

if __name__ == '__main__':
    print('This is the diff module')

    # Load th two images
    path_folder = 'v2/imgs/'
    path_img = os.path.join(path_folder, 'img_new3_undistorted.jpg')
    path_old = os.path.join(path_folder, 'img_old4.jpg')
    img = cv2.imread(path_img)
    img_old = cv2.imread(path_old)

    gray_new = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_old = cv2.cvtColor(img_old, cv2.COLOR_BGR2GRAY)

    target = cv2.bilateralFilter(gray_old, 9, 75, 75)
    img_new = cv2.bilateralFilter(gray_new, 9, 75, 75)

    # Detect SIFT features and compute descriptors
    path = './v2/output/img_new3_undistorted_img_old4_matches.npz'
    npz_c1_old = np.load(path)

    kp_c1_c1old_ = npz_c1_old['keypoints0']
    kp_old_c1old_ = npz_c1_old['keypoints1']
    kp_c1_c1old = np.float32([kp_c1_c1old_[i] for i,x in enumerate(npz_c1_old['matches']) if x != -1])
    kp_old_c1old = np.float32([kp_old_c1old_[x] for i,x in enumerate(npz_c1_old['matches']) if x != -1])

    M, mask = cv2.findHomography(kp_c1_c1old, kp_old_c1old, cv2.RANSAC, 5.0)

    h, w = target.shape[:2]

    im_dst = cv2.warpPerspective(img_new, M, (w, h))

    thresh_old = cv2.adaptiveThreshold(target, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    thresh_new = cv2.adaptiveThreshold(im_dst, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    thresh_new = thresh_new.astype(np.int8)
    thresh_old = thresh_old.astype(np.int8)

    diff = (thresh_new - thresh_old)
    seg_labels = np.where(diff < 0, 2, diff)

    colours = np.array([[0, 0, 0], [255, 0, 0], [0, 255, 0]])

    fig, ax = plt.subplots(1, 3)
    ax[0].set_title("target otsu binarized")
    ax[0].imshow(thresh_old, cmap='hot')
    ax[1].set_title("warped new otsu binarized")
    ax[1].imshow(thresh_new, cmap='hot')
    ax[2].set_title("diff")
    ax[2].imshow(colours[seg_labels], cmap='hot')
    plt.show()