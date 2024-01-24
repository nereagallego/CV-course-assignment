# Computer Vision Course Assignment
# Author: Nerea Gallego (801950)

import cv2
import numpy as np
import matplotlib.pyplot as plt

import scipy as sc
import scipy.optimize as scOptim

def draw3DLine(ax, xIni, xEnd, strStyle, lColor, lWidth):
    """
    Draw a segment in a 3D plot
    -input:
        ax: axis handle
        xIni: Initial 3D point.
        xEnd: Final 3D point.
        strStyle: Line style.
        lColor: Line color.
        lWidth: Line width.
    """
    ax.plot([np.squeeze(xIni[0]), np.squeeze(xEnd[0])], [np.squeeze(xIni[1]), np.squeeze(xEnd[1])], [np.squeeze(xIni[2]), np.squeeze(xEnd[2])],
            strStyle, color=lColor, linewidth=lWidth)

def drawRefSystem(ax, T_w_c, strStyle, nameStr):
    """
        Draw a reference system in a 3D plot: Red for X axis, Green for Y axis, and Blue for Z axis
    -input:
        ax: axis handle
        T_w_c: (4x4 matrix) Reference system C seen from W.
        strStyle: lines style.
        nameStr: Name of the reference system.
    """
    draw3DLine(ax, T_w_c[0:3, 3:4], T_w_c[0:3, 3:4] + T_w_c[0:3, 0:1], strStyle, 'r', 1)
    draw3DLine(ax, T_w_c[0:3, 3:4], T_w_c[0:3, 3:4] + T_w_c[0:3, 1:2], strStyle, 'g', 1)
    draw3DLine(ax, T_w_c[0:3, 3:4], T_w_c[0:3, 3:4] + T_w_c[0:3, 2:3], strStyle, 'b', 1)
    ax.text(np.squeeze( T_w_c[0, 3]+0.1), np.squeeze( T_w_c[1, 3]+0.1), np.squeeze( T_w_c[2, 3]+0.1), nameStr)

# Funtion to compute the intersectio between two sets of matches
def intersection(k1, k1_2, k2, k3):
    print(k1.shape, k1_2.shape, k2.shape, k3.shape)
    k2_n = []
    k3_n = []
    k1_n = []
    i = 0
    for k in k1:
        j = 0
        for k2_ in k1_2:
            if np.linalg.norm(np.array(k) - np.array(k2_)) <= 2:    
                k2_n.append(k2[i,:])
                k1_n.append(k)
                k3_n.append(k3[j,:])
            j += 1
        i += 1
    return np.array(k1_n), np.array(k2_n), np.array(k3_n)

def normalizationMatrix(nx,ny):
    """
    Estimation of fundamental matrix(F) by matched n matched points.
    n >= 8 to assure the algorithm.

    -input:
        nx: number of columns of the matrix
        ny: number of rows of the matrix
    -output:
        Nv: normalization matrix such that xN = Nv @ x
    """
    Nv = np.array([[1/nx, 0, -1/2], [0, 1/ny, -1/2], [0, 0, 1]])
    return Nv

def compute_fundamental_matrix(points1, points2, nx1, ny1, nx2, ny2):
    # Normalize the points
    N1 = normalizationMatrix(nx1, ny1)
    N2 = normalizationMatrix(nx2, ny2)
    points1 = N1 @ points1
    points2 = N2 @ points2
    # Compute the fundamental matrix

    A = np.zeros((points1.shape[1], 9))
    for i in range(points1.shape[1]):
        A[i, :] = [points1[0, i] * points2[0, i], points2[0, i] * points1[1, i], points2[0, i], points1[0, i] * points2[1, i], points1[1, i] * points2[1, i], points2[1,i], points1[0,i], points1[1,i], 1]
    
    _, _, V = np.linalg.svd(A)

    # compute the fundamental matrix from the right singular vector corresponding to the smallest singular value
    F = V[-1, :].reshape((3, 3))
    U, S, V = np.linalg.svd(F)

    # enforce rank 2 constraint
    S[2] = 0
    F = np.dot(U, np.dot(np.diag(S), V))
    F = np.dot(N2.T, np.dot(F, N1))
    return F/F[2,2]

# Function to compute the fundamental matrix with RANSAC
def calculate_RANSAC_own_F(source,dst,third, threshold, nx1, ny1, nx2, ny2):
    num_samples = 8
    spFrac = 0.6  # spurious fraction
    P = 0.999  # probability of selecting at least one sample without spurious

    # number m of random samples
    nAttempts = np.round(np.log(1 - P) / np.log(1 - np.power((1 - spFrac),num_samples)))
    num_attempts = nAttempts.astype(int)
    num_attempts = 5000

    source = np.vstack((source, np.ones((1, source.shape[1]))))
    dst = np.vstack((dst, np.ones((1, dst.shape[1]))))
    third = np.vstack((third, np.ones((1, third.shape[1]))))
    print(source.shape, dst.shape, third.shape)
    matches = np.vstack((source,dst, third))
    best_model_votes = 0
    best_model_matches = None

    for kAttempt in range(num_attempts):
        votes = 0

        rng = np.random.default_rng()
        indx_subset = rng.choice(matches.shape[1] - 1, size=num_samples, replace=False)
        matches_subset = []
        rest_matches = []
        good_matches = []

        for i in range(matches.shape[1]):
            if i in indx_subset:
                matches_subset.append(matches[:, i])
            else:
                rest_matches.append(matches[:, i])

        matches_subset = np.array(matches_subset).T
        rest_matches = np.array(rest_matches).T

        F = compute_fundamental_matrix(matches_subset[0:3,:],matches_subset[3:6,:], nx1, ny1, nx2, ny2)
        # F = compute_fundamental_matrix(matches_subset[0:3,:],matches_subset[3:6,:])
        if F is not None:
            for i in range(rest_matches.shape[1]):

                x1 = rest_matches[0:3, i]
                x2 = rest_matches[3:6, i]

                l_2 = F @ x1
                    
                dist_x2_l2 = np.abs(np.dot(x2.T,np.dot(F , x1))/ np.sqrt((l_2[0]**2 + l_2[1]**2)))

                if dist_x2_l2 < threshold:
                    good_matches.append(rest_matches[:, i])
                    votes = votes + 1

            if votes > best_model_votes:
                best_model_votes = votes
                print(votes)
                F_most_voted = F
                best_model_matches = np.hstack((matches_subset, np.array(good_matches).T))


    return F_most_voted, best_model_matches

# Function to compute the essential matrix
def compute_essential_matrix(F, K):
    E = K.T @ F @ K
    return E

# Triangulate a set of points given the projection matrices of two cameras.
def triangulation(P1, P2, points1, points2):    
    points3D = np.zeros((4, points1.shape[1]))
    for i in range(points1.shape[1]):
        p1 = points1[:, i]
        p2 = points2[:, i]
        # A = [p1[0] * P1[2, :] - P1[0, :], p1[1] * P1[2, :] - P1[1, :], p2[0] * P2[2, :] - P2[0, :], p2[1] * P2[2, :] - P2[1, :]]
        A = np.vstack((p1[0] * P1[2, :] - P1[0, :], 
                       p1[1] * P1[2, :] - P1[1, :], 
                       p2[0] * P2[2, :] - P2[0, :], 
                       p2[1] * P2[2, :] - P2[1, :]))
        _, _, V = np.linalg.svd(A)
        X = V[-1, :]
        points3D[:, i] = X / X[3]

    return points3D

def points_in_front_of_both_cameras(x1, x2, T, K):
    I = ensamble_T(np.diag((1, 1, 1)), np.zeros((3)))[0:3]

    P1 = K @ I
    P2 = K @ T[0:3]

    points3d = triangulation(P1, P2, x1, x2)
    points3d = points3d.T

    in_front = 0

    points_front = []
    for point in points3d:
        # if point[2] < 0:
            # continue

        # z > 0 in C1 frame
        # pointFrame = T @ point.reshape(-1,1)
        # if pointFrame[2] > 0:
        R = T[0:3, 0:3]
        t = T[0:3, 3]
        if point[2] > 0 and np.dot(R[2], point[0:3] - t) > 0:
            in_front += 1
            points_front.append(point)
    
    return in_front, points_front

def ensamble_T(R_w_c, t_w_c) -> np.array:
    """
    Ensamble the a SE(3) matrix with the rotation matrix and translation vector.
    """
    T_w_c = np.zeros((4, 4))
    T_w_c[0:3, 0:3] = R_w_c
    T_w_c[0:3, 3] = t_w_c
    T_w_c[3, 3] = 1
    return T_w_c

# Decompose the Essential matrix
def decompose_essential_matrix(x1, x2, E, K, idx=None):
    # Compute the SVD of the essential matrix
    U, _, V = np.linalg.svd(E)
    t = U[:,2]
    
    # Ensure that the determinant of U and Vt is positive (to ensure proper rotation)

    W = np.array([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1]
    ])

    R_90 = U @ W @ V
    R_n90 = U @ W.T @ V 

    # Compute the four possible solutions
    solutions = []
    for R in [R_90, R_n90]:
        for i in [t, -t]:
            if np.linalg.det(R) < 0:
                R *= -1
                i *= -1
                solutions.append(ensamble_T(R, i))

    points_front = []
    points = []
    for T in solutions:
        v1, v2 = points_in_front_of_both_cameras(x1, x2, T, K)
        points_front.append(v1)
        points.append(v2)
    T = solutions[np.argmax(points_front)]
    if idx is None:
        return T, np.array(points[np.argmax(points_front)])
    else:
        return solutions[idx], np.array(points[idx])

if __name__ == '__main__':
    Kc_new = np.loadtxt('calibration_matrix.txt')


    path_image_new_1 = 'imgs1/img_new1.jpg'
    path_image_new_2 = 'imgs1/img_new3.jpg'
    path_image_old = 'imgs1/img_old.jpg'

    img1 = cv2.imread(path_image_new_1)
    img2 = cv2.imread(path_image_new_2)
    img_old = cv2.imread(path_image_old)

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

    print(keypoints_new_1.shape, keypoints_new_2.shape, keypoints_old.shape)

    F, matches = calculate_RANSAC_own_F(keypoints_new_1.T, keypoints_new_2.T, keypoints_old.T, 2, img1.shape[1], img1.shape[0], img2.shape[1], img2.shape[0])

    kp_new1 = matches[0:3,:].T
    kp_new2 = matches[3:6,:].T
    kp_old = matches[6:9,:].T

    print(kp_new1.shape, kp_new2.shape, kp_old.shape)

    E = compute_essential_matrix(F, Kc_new)

    Rt, X_3d = decompose_essential_matrix(kp_new1[:,0:2].T, kp_new2[:,0:2].T, E, Kc_new)
    print(X_3d.shape)


    # Plot the 3D points
    fig = plt.figure()
    ax = plt.axes(projection='3d', adjustable='box')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    drawRefSystem(ax, np.eye(4), '-', 'C1')
    drawRefSystem(ax, np.eye(4) @ np.linalg.inv(Rt) , '-', 'C2')
    ax.scatter(X_3d[:,0], X_3d[:,1], X_3d[:,2], marker='.')
    plt.show()
