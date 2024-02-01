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


    path_image_new_1 = 'imgs1/img_new1_undistorted.jpg'
    path_image_new_2 = 'imgs1/img_new3_undistorted.jpg'
    path_image_old = 'imgs1/img_old.jpg'

    img1 = cv2.imread(path_image_new_1)
    img2 = cv2.imread(path_image_new_2)
    img_old = cv2.imread(path_image_old)

    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    img_old = cv2.cvtColor(img_old, cv2.COLOR_BGR2RGB)

    print('Imgs loaded')

    path = './output/img_new1_undistorted_img_new3_undistorted_matches.npz'
    npz = np.load(path)
    keypoints_SG_0_new = npz['keypoints0']
    keypoints_SG_1_new = npz['keypoints1']
    path2 = './output/img_new1_undistorted_img_old_matches.npz'
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


    # F, matches = calculate_RANSAC_own_F(keypoints_new_1.T, keypoints_new_2.T, keypoints_old.T, 2, img1.shape[1], img1.shape[0], img2.shape[1], img2.shape[0])
    # np.savetxt('F.txt', F)
    # np.savetxt('good_matches.txt', matches)
    F = np.loadtxt('F.txt')
    matches = np.loadtxt('good_matches.txt')

    kp_new1 = matches[0:3,:].T
    kp_new2 = matches[3:6,:].T
    kp_old = matches[6:9,:].T

    T_w_c1 = ensamble_T(np.diag((1, 1, 1)), np.zeros((3)))

    T_c2_c1, X_3d = sfm(F, Kc_new, kp_new1[:,0:2].T, kp_new2[:,0:2].T, kp_old[:,0:2].T)

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
    ax[0].set_title('Residuals before Bundle adjustment Image1')
    plotResidual2(kp_new1, x1_p.T, 'k-', ax[0])
    ax[1].imshow(img2, cmap='gray', vmin=0, vmax=255)
    ax[1].set_title('Residuals before Bundle adjustment Image2')
    plotResidual2(kp_new2, x2_p.T, 'k-', ax[1])
    plt.show()

    R = T_c2_c1[0:3, 0:3]
    t = T_c2_c1[0:3, 3].reshape(-1,1)

    elevation = np.arccos(t[2])
    azimuth = np.arctan2(t[1], t[0])

    Op = [elevation, azimuth] + crossMatrixInv(sc.linalg.logm(R)) + X_3d[:,0:3].flatten().tolist()

    # OpOptim = scOptim.least_squares(resBundleProjection, Op, args=(kp_new1[:,0:2].T, kp_new2[:,0:2].T, Kc_new, kp_new1.shape[0]), method='trf', loss='huber', verbose=2)
    # OpOptim = OpOptim.x
    # np.savetxt('Optimization.txt', OpOptim)
    OpOptim = np.loadtxt('Optimization.txt')



    R_c2_c1 = sc.linalg.expm(crossMatrix(OpOptim[2:5]))
    t_c2_c1 = np.array([np.sin(OpOptim[0])*np.cos(OpOptim[1]), np.sin(OpOptim[0])*np.sin(OpOptim[1]), np.cos(OpOptim[0])])
    T_c2_c1 = ensamble_T(R_c2_c1, t_c2_c1)
    points_3d = np.concatenate((OpOptim[5:8], np.array([1.0])), axis=0)
    for i in range(X_3d.shape[0]-1):
        points_3d = np.vstack((points_3d, np.concatenate((OpOptim[8+3*i: 8+3*i+3], np.array([1.0])) ,axis=0)))

    # Plot the 3D points
    fig = plt.figure()
    ax = plt.axes(projection='3d', adjustable='box')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    drawRefSystem(ax, T_w_c1, '-', 'C1')
    drawRefSystem(ax, T_w_c1 @ np.linalg.inv(T_c2_c1) , '-', 'C2')
    points_3d = T_w_c1 @ (points_3d).T
    points_3d = points_3d.T
    ax.scatter(points_3d[:,0], points_3d[:,1], points_3d[:,2], marker='.')
    xFakeBoundingBox = np.linspace(0, 4, 2)
    yFakeBoundingBox = np.linspace(0, 4, 2)
    zFakeBoundingBox = np.linspace(0, 4, 2)
    ax.plot(xFakeBoundingBox, yFakeBoundingBox, zFakeBoundingBox, 'w.')
    plt.show()

    P1 = aux_matrix @ T_w_c1
    P2 = aux_matrix @ (T_w_c1 @ T_c2_c1)

    x1_p = P1 @ points_3d.T
    x1_p = x1_p / x1_p[2, :]
    x2_p = P2 @ points_3d.T
    x2_p = x2_p / x2_p[2, :]

    fig, ax = plt.subplots(1,2, figsize=(10,5))
    ax[0].imshow(img1)
    ax[0].set_title('Residuals after Bundle adjustment Image1')
    plotResidual2(kp_new1, x1_p.T, 'k-', ax[0])
    ax[1].imshow(img2)
    ax[1].set_title('Residuals after Bundle adjustment Image2')
    plotResidual2(kp_new2, x2_p.T, 'k-', ax[1])
    plt.show()

    P_old = DLTcamera(kp_old, points_3d)
    print('P shape ', P_old.shape)

    M = P_old[0:3,0:3]
    [K_old, R_c3_c1, t_c3_c1] = cv2.decomposeProjectionMatrix(np.sign(np.linalg.det(M)) * P_old)[:3]

    t_c3_c1 = (t_c3_c1[:3] / t_c3_c1[3]).reshape((3,))

    T_c3_c1 = ensamble_T(R_c3_c1, t_c3_c1)

    fig = plt.figure()
    ax = plt.axes(projection='3d', adjustable='box')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    drawRefSystem(ax, T_w_c1, '-', 'C1')
    drawRefSystem(ax, T_w_c1 @ np.linalg.inv(T_c2_c1) , '-', 'C2')
    drawRefSystem(ax, T_w_c1 @ np.linalg.inv(T_c3_c1) , '-', 'C old')
    ax.scatter(points_3d[:,0], points_3d[:,1], points_3d[:,2], marker='.')
    # plotNumbered3DPoints(ax, points_3d.T, 'r', 0.1)
    xFakeBoundingBox = np.linspace(0, 4, 2)
    yFakeBoundingBox = np.linspace(0, 4, 2)
    zFakeBoundingBox = np.linspace(0, 4, 2)
    ax.plot(xFakeBoundingBox, yFakeBoundingBox, zFakeBoundingBox, 'w.')
    plt.show()

    P1 = aux_matrix @ T_w_c1
    P2 = aux_matrix @ (T_w_c1 @ T_c2_c1)
    P_old = aux_matrix @ (T_w_c1 @ T_c3_c1)

    x1_p = P1 @ points_3d.T
    x1_p = x1_p / x1_p[2, :]
    x2_p = P2 @ points_3d.T
    x2_p = x2_p / x2_p[2, :]
    x3_p = P_old @ points_3d.T

    fig, ax = plt.subplots(1,3, figsize=(10,5))
    ax[0].imshow(img1)
    ax[0].set_title('Residuals after Bundle adjustment Image1')
    # plotResidual2(kp_new1, x1_p.T, 'k-', ax[0])
    plotNumberedImagePoints(kp_new1[:,0:2].T, 'r',4, ax[0])
    ax[1].imshow(img2)
    ax[1].set_title('Residuals after Bundle adjustment Image2')
    # plotResidual2(kp_new2, x2_p.T, 'k-', ax[1])
    plotNumberedImagePoints(kp_new2[:,0:2].T, 'r',4, ax[1])
    ax[2].imshow(img_old)
    # plotResidual2(kp_old, x3_p.T, 'k-', ax[2])
    plotNumberedImagePoints(kp_old[:,0:2].T, 'r',4, ax[2])
    plt.show()