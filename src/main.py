# Computer Vision Course Assignment
# Author: Nerea Gallego (801950)

import cv2
import numpy as np
import matplotlib.pyplot as plt

import scipy as sc
import scipy.optimize as scOptim
import os

from utils.utils import *
from utils.functions import *
from utils.plot import *

if __name__ == '__main__':
    plot_flag = True
    reconstruction_flag = True
    differeces_flag = False

    Kc = np.loadtxt('Kc.txt')

    # Load images
    path_folder = 'v2/imgs/'
    path_img1 = os.path.join(path_folder, 'img_new3_undistorted.jpg')
    # path_img2 = os.path.join(path_folder, 'img_new2_undistorted.jpg')
    path_img3 = os.path.join(path_folder, 'img_new4_undistorted.jpg')
    # path_img4 = os.path.join(path_folder, 'img_new4_undistorted.jpg')
    path_old = os.path.join(path_folder, 'img_old4.jpg')

    img1 = cv2.imread(path_img1)
    # img2 = cv2.imread(path_img2)
    img3 = cv2.imread(path_img3)
    # img4 = cv2.imread(path_img4)
    img_old = cv2.imread(path_old)

    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
    # img4 = cv2.cvtColor(img4, cv2.COLOR_BGR2RGB)
    img_old = cv2.cvtColor(img_old, cv2.COLOR_BGR2RGB)


    # Load matches
    # path = './output/img_new1_undistorted_img_new2_undistorted_matches.npz'
    # npz_c1_c2 = np.load(path)
    path = './v2/output/img_new3_undistorted_img_new4_undistorted_matches.npz'
    npz_c1_c3 = np.load(path)
    # path = './output/img_new1_undistorted_img_new4_undistorted_matches.npz'
    # npz_c1_c4 = np.load(path)
    path = './v2/output/img_new3_undistorted_img_old4_matches.npz'
    npz_c1_old = np.load(path)


    # Get matches
    # kp_c1_c1c2_ = npz_c1_c2['keypoints0']
    # kp_c2_c1c2_ = npz_c1_c2['keypoints1']
    kp_c1_c1c3_ = npz_c1_c3['keypoints0']
    kp_c3_c1c3_ = npz_c1_c3['keypoints1']
    # kp_c1_c1c4_ = npz_c1_c4['keypoints0']
    # kp_c4_c1c4_ = npz_c1_c4['keypoints1']
    kp_c1_c1old_ = npz_c1_old['keypoints0']
    kp_old_c1old_ = npz_c1_old['keypoints1']

    # kp_c1_c1c2 = np.float32([kp_c1_c1c2_[i] for i,x in enumerate(npz_c1_c2['matches']) if x != -1])
    # kp_c2_c1c2 = np.float32([kp_c2_c1c2_[x] for i,x in enumerate(npz_c1_c2['matches']) if x != -1])
    kp_c1_c1c3 = np.float32([kp_c1_c1c3_[i] for i,x in enumerate(npz_c1_c3['matches']) if x != -1])
    kp_c3_c1c3 = np.float32([kp_c3_c1c3_[x] for i,x in enumerate(npz_c1_c3['matches']) if x != -1])
    # kp_c1_c1c4 = np.float32([kp_c1_c1c4_[i] for i,x in enumerate(npz_c1_c4['matches']) if x != -1])
    # kp_c4_c1c4 = np.float32([kp_c4_c1c4_[x] for i,x in enumerate(npz_c1_c4['matches']) if x != -1])
    kp_c1_c1old = np.float32([kp_c1_c1old_[i] for i,x in enumerate(npz_c1_old['matches']) if x != -1])
    kp_old_c1old = np.float32([kp_old_c1old_[x] for i,x in enumerate(npz_c1_old['matches']) if x != -1])

    kp_c1, kp_c3, kp_old = intersection(kp_c1_c1c3, kp_c1_c1old, kp_c3_c1c3, kp_old_c1old)
    print('Number of matches after intersection: ', kp_c1.shape[0], kp_c3.shape[0], kp_old.shape[0])

    F, mask = cv2.findFundamentalMat(kp_c1, kp_c3, cv2.RANSAC, ransacReprojThreshold=1.0)
    print(mask.ravel())
    kp_c1 = kp_c1[mask.ravel() == 1]
    kp_c3 = kp_c3[mask.ravel() == 1]
    kp_old = kp_old[mask.ravel() == 1]

    if plot_flag:
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].imshow(img1)
        ax[0].set_title('Matches C1')
        ax[0].scatter(kp_c1[:,0], kp_c1[:,1], c='r', marker='.')

        ax[1].imshow(img3)
        ax[1].set_title('Matches C3')
        ax[1].scatter(kp_c3[:,0], kp_c3[:,1], c='r', marker='.')

        ax[2].imshow(img_old)
        ax[2].set_title('Matches Old')
        ax[2].scatter(kp_old[:,0], kp_old[:,1], c='r', marker='.')

        plt.show()

    print('Number of matches after RANSAC: ', kp_c1.shape[0], kp_c3.shape[0], kp_old.shape[0])

    T_w_c1 = ensamble_T(np.diag((1,1,1)), np.array([0,0,0]))

    T_c3_c1, X3d, mask = sfm(F, Kc, kp_c1[:, :2].T, kp_c3[:, :2].T, kp_old[:, :2].T)
    kp_c1 = kp_c1[mask]
    kp_c3 = kp_c3[mask]
    kp_old = kp_old[mask]

    print('Number of matches after SfM: ', kp_c1.shape[0], kp_c3.shape[0], kp_old.shape[0])

    if plot_flag: 
        # Plot 3D points
        fig = plt.figure()
        ax = plt.axes(projection='3d', adjustable='box')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        drawRefSystem(ax, T_w_c1, '-', 'C1')
        T_w_c3 = T_w_c1 @ np.linalg.inv(T_c3_c1)
        drawRefSystem(ax, T_w_c3, '-', 'C3')
        ax.scatter(X3d[:,0], X3d[:,1], X3d[:,2], c='r', marker='.')
        axisEqual3D(ax)
        plt.show()

    idem = np.hstack((np.identity(3), np.zeros(3).reshape(3,1)))
    aux = Kc @ idem
    P3 = aux @ T_c3_c1

    P1 = aux @ T_w_c1

    x1 = P1 @ X3d.T
    x1 = x1 / x1[2,:]  
    x3 = P3 @ X3d.T
    x3 = x3 / x3[2,:]

    if plot_flag:

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(img1, cmap='gray', vmin=0, vmax=255)
        ax[0].set_title('Residuals before BA')
        plotResidual2(kp_c1, x1.T, 'k-', ax[0])

        ax[1].imshow(img3, cmap='gray', vmin=0, vmax=255)
        ax[1].set_title('Residuals before BA')
        plotResidual2(kp_c3, x3.T, 'k-', ax[1])
        plt.show()

        res = 0
        res += sum(sum(abs(kp_c1 - x1[:2].T)))
        res += sum(sum(abs(kp_c3 - x3[:2].T)))
        print('Total residuals before BA: ', res/(2*len(kp_c1)))

    R = T_c3_c1[:3,:3]
    t = T_c3_c1[:3,3].reshape(3,1)

    print(crossMatrixInv(sc.linalg.logm(R)))

    Op = t.flatten().tolist() + crossMatrixInv(sc.linalg.logm(R)).flatten().tolist() + X3d[:,:3].flatten().tolist()

    OpOptim = scOptim.least_squares(resBundleProjection, Op, args=(kp_c1.T, kp_c3.T, Kc, kp_c1.shape[0]), method='trf', loss='huber', verbose=2)

    OpOptim = OpOptim.x

    R_c3_c1 = sc.linalg.expm(crossMatrix(OpOptim[3:6]))
    t_c3_c1 = np.array([OpOptim[0], OpOptim[1], OpOptim[2]])
    T_c3_c1 = ensamble_T(R_c3_c1, t_c3_c1)
    X3d_optim = np.concatenate((OpOptim[6:9], np.array([1.0])), axis=0)
    for i in range(X3d.shape[0]-1):
        X3d_optim = np.vstack((X3d_optim, np.concatenate((OpOptim[9+3*i:12+3*i], np.array([1.0])), axis=0)))

    if plot_flag:
        fig = plt.figure()
        ax = plt.axes(projection='3d', adjustable='box')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        drawRefSystem(ax, T_w_c1, '-', 'C1')
        T_w_c3 = T_w_c1 @ np.linalg.inv(T_c3_c1)
        drawRefSystem(ax, T_w_c3, '-', 'C3')
        ax.scatter(X3d_optim[:,0], X3d_optim[:,1], X3d_optim[:,2], c='r', marker='.')
        axisEqual3D(ax)
        plt.show()

    P3 = aux @ T_c3_c1

    x1 = P1 @ X3d_optim.T
    x1 = x1 / x1[2,:]
    x3 = P3 @ X3d_optim.T
    x3 = x3 / x3[2,:]


    if plot_flag:
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(img1, cmap='gray', vmin=0, vmax=255)
        ax[0].set_title('Residuals after BA')
        plotResidual2(kp_c1[mask], x1.T, 'k-', ax[0])

        ax[1].imshow(img3, cmap='gray', vmin=0, vmax=255)
        ax[1].set_title('Residuals after BA')
        plotResidual2(kp_c3[mask], x3.T, 'k-', ax[1])
        plt.show()

        res = 0
        res += sum(sum(abs(kp_c1 - x1[:2].T)))
        res += sum(sum(abs(kp_c3 - x3[:2].T)))
        print('Total residuals before BA: ', res/(2*len(kp_c1)))

    Pold = DLTcamera(kp_old, X3d_optim)

    xold = Pold @ X3d_optim.T
    xold = xold / xold[2,:]

    if plot_flag:
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].imshow(img1)
        ax[0].set_title('Residuals C1')
        plotResidual2(kp_c1, x1.T, 'k-', ax[0])

        ax[1].imshow(img3)
        ax[1].set_title('Residuals C3')
        plotResidual2(kp_c3, x3.T, 'k-', ax[1])

        ax[2].imshow(img_old)
        ax[2].set_title('Residuals Old')
        plotResidual2(kp_old, xold.T, 'k-', ax[2])

        plt.show()

        res = 0
        res += sum(sum(abs(kp_c1 - x1[:2].T)))
        res += sum(sum(abs(kp_c3 - x3[:2].T)))
        res += sum(sum(abs(kp_old - xold[:2].T)))
        print('Total residuals before BA: ', res/(3*len(kp_c1)))

    Kold, R_c1_cold, t_c1_cold = decomposeP(Pold)
    T_w_cold = ensamble_T(R_c1_cold, t_c1_cold)
    # T_c1_cold = np.linalg.inv(T_cold_c1)

    if plot_flag:
        fig = plt.figure()
        ax = plt.axes(projection='3d', adjustable='box')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        drawRefSystem(ax, T_w_c1, '-', 'C1')
        T_w_c3 = T_w_c1 @ np.linalg.inv(T_c3_c1)
        drawRefSystem(ax, T_w_c3, '-', 'C3')
        # T_w_old = T_w_c1 @ T_c1_cold
        drawRefSystem(ax, T_w_cold, '-', 'Old')
        # drawRefSystem(ax, np.linalg.inv(T_w_cold), '-', 'Old2')
        ax.scatter(X3d_optim[:,0], X3d_optim[:,1], X3d_optim[:,2], c='r', marker='.')
        axisEqual3D(ax)
        plt.show()

        print('Pose C1: {}'.format(T_w_c1))
        print('Pose C3: {}'.format(T_w_c3))
        print('Pose Old: {}'.format(T_w_cold))

    Pold_optim = refinePoseOldCamera(X3d_optim, kp_old, Pold)
    Kold_oltim, R_cold_c1_optim, t_cold_c1_optim = decomposeP(Pold_optim)
    T_w_cold_optim = ensamble_T(R_cold_c1_optim, t_cold_c1_optim)


    if plot_flag:
        fig = plt.figure()
        ax = plt.axes(projection='3d', adjustable='box')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        drawRefSystem(ax, T_w_c1, '-', 'C1')
        T_w_c3 = T_w_c1 @ np.linalg.inv(T_c3_c1)
        drawRefSystem(ax, T_w_c3, '-', 'C3')
        # T_w_old = T_w_c1 @ T_c1_cold
        drawRefSystem(ax, T_w_cold_optim, '-', 'Old')
        # drawRefSystem(ax, np.linalg.inv(T_w_cold_optim), '-', 'Old2')
        ax.scatter(X3d_optim[:,0], X3d_optim[:,1], X3d_optim[:,2], c='r', marker='.')
        axisEqual3D(ax)
        plt.show()

    xold = Pold_optim @ X3d_optim.T
    xold = xold / xold[2,:]

    if plot_flag:
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].imshow(img1)
        ax[0].set_title('Residuals C1')
        plotResidual2(kp_c1, x1.T, 'k-', ax[0])

        ax[1].imshow(img3)
        ax[1].set_title('Residuals C3')
        plotResidual2(kp_c3, x3.T, 'k-', ax[1])

        ax[2].imshow(img_old)
        ax[2].set_title('Residuals Old')
        plotResidual2(kp_old, xold.T, 'k-', ax[2])

        plt.show()

        res = 0
        res += sum(sum(abs(kp_c1 - x1[:2].T)))
        res += sum(sum(abs(kp_c3 - x3[:2].T)))
        res += sum(sum(abs(kp_old - xold[:2].T)))
        print('Total residuals before BA: ', res/(3*len(kp_c1)))
    
