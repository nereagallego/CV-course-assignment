# Computer Vision Course Assignment
# Author: Nerea Gallego (801950)

import cv2
import numpy as np
import matplotlib.pyplot as plt

import scipy as sc
import scipy.optimize as scOptim

from utils.utils import *
from utils.functions import *
from utils.plot import *

if __name__ == '__main__':
    Kc_new = np.loadtxt('calibration_matrix.txt')


    path_image_new_1 = 'imgs1/img_new1.jpg'
    path_image_new_2 = 'imgs1/img_new3.jpg'
    path_image_old = 'imgs1/img_old.jpg'

    img1 = cv2.imread(path_image_new_1)
    img2 = cv2.imread(path_image_new_2)
    img_old = cv2.imread(path_image_old)

    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    img_old = cv2.cvtColor(img_old, cv2.COLOR_BGR2RGB)

    print('Imgs loaded')

    path = './output/img_new1_img_new3_matches.npz'
    npz = np.load(path)
    keypoints_SG_0_new = npz['keypoints0']
    keypoints_SG_1_new = npz['keypoints1']
    path2 = './output/img_new1_img_old_matches.npz'
    npz2 = np.load(path2)
    keypoints_SG_0_old = npz2['keypoints0']
    keypoints_SG_1_old = npz2['keypoints1']

    matchesListSG_0 = [i for i, x in enumerate(npz['matches']) if x != -1]
    matchesListSG_1 = [x for i, x in enumerate(npz['matches']) if x != -1]
    matchesListSG_0_old = [i for i, x in enumerate(npz2['matches']) if x != -1]
    matchesListSG_1_old = [x for i, x in enumerate(npz2['matches']) if x != -1]

    # Matched points from SuperGlue
    srcPts_SG = np.float32([keypoints_SG_0_new[m] for m in matchesListSG_0])
    dstPts_SG = np.float32([keypoints_SG_1_new[m] for m in matchesListSG_1])
    srcPts_SG_old = np.float32([keypoints_SG_0_old[m] for m in matchesListSG_0_old])
    dstPts_SG_old = np.float32([keypoints_SG_1_old[m] for m in matchesListSG_1_old])

    keypoints_new_1, keypoints_new_2, keypoints_old = intersection(srcPts_SG, srcPts_SG_old, dstPts_SG, dstPts_SG_old)


    F, matches = calculate_RANSAC_own_F(keypoints_new_1.T, keypoints_new_2.T, keypoints_old.T, 2, img1.shape[1], img1.shape[0], img2.shape[1], img2.shape[0])

    kp_new1 = matches[0:3,:].T
    kp_new2 = matches[3:6,:].T
    kp_old = matches[6:9,:].T

    T_w_c1 = ensamble_T(np.diag((1, 1, 1)), np.zeros((3)))

    T_c2_c1, X_3d = sfm(F, Kc_new, kp_new1[:,0:2].T, kp_new2[:,0:2].T)

    # Plot the 3D points
    fig = plt.figure()
    ax = plt.axes(projection='3d', adjustable='box')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    drawRefSystem(ax, T_w_c1, '-', 'C1')
    T_w_c2 = T_w_c1 @ np.linalg.inv(T_c2_c1)
    drawRefSystem(ax, T_w_c2 , '-', 'C2')
    ax.scatter(X_3d[:,0], X_3d[:,1], X_3d[:,2], marker='.')
    xFakeBoundingBox = np.linspace(0, 4, 2)
    yFakeBoundingBox = np.linspace(0, 4, 2)
    zFakeBoundingBox = np.linspace(0, 4, 2)
    ax.plot(xFakeBoundingBox, yFakeBoundingBox, zFakeBoundingBox, 'w.')
    plt.show()

    idem = np.hstack((np.identity(3), np.zeros(3).reshape(3,1)))
    aux_matrix = np.dot(Kc_new,idem)
    P2 = aux_matrix @ np.linalg.inv(T_w_c2)

    P1_est = Kc_new @ idem
    x1_p = P1_est @ X_3d.T
    x1_p = x1_p / x1_p[2, :]
    x2_p = P2 @ X_3d.T
    x2_p = x2_p / x2_p[2, :]

    fig, ax = plt.subplots(1,2, figsize=(10,5))
    ax[0].imshow(img1, cmap='gray', vmin=0, vmax=255)
    ax[0].set_title('Residuals after Bundle adjustment Image1')
    plotResidual2(kp_new1, x1_p.T, 'k-', ax[0])
    ax[1].imshow(img2, cmap='gray', vmin=0, vmax=255)
    ax[1].set_title('Residuals after Bundle adjustment Image2')
    plotResidual2(kp_new2, x2_p.T, 'k-', ax[1])
    plt.show()