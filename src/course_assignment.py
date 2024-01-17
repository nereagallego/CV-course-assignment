# Computer Vision Course Assignment
# Author: Nerea Gallego (801950)

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

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

def matchWith2NDRR(desc1, desc2, distRatio, minDist):
    """
    Nearest Neighbours Matching algorithm checking the Distance Ratio.
    A match is accepted only if its distance is less than distRatio times
    the distance to the second match.
    -input:
        desc1: descriptors from image 1 nDesc x 128
        desc2: descriptors from image 2 nDesc x 128
        distRatio:
    -output:
       matches: nMatches x 3 --> [[indexDesc1,indexDesc2,descriptorDistance],...]]
    """
    matches = []
    nDesc1 = desc1.shape[0]
    for kDesc1 in range(nDesc1):
        dist = np.sqrt(np.sum((desc2 - desc1[kDesc1, :]) ** 2, axis=1))
        indexSort = np.argsort(dist)
        if (dist[indexSort[0]] < minDist):
            matches.append([kDesc1, indexSort[0], dist[indexSort[0]]])
    return matches

def indexMatrixToMatchesList(matchesList):
    """
     -input:
         matchesList: nMatches x 3 --> [[indexDesc1,indexDesc2,descriptorDistance],...]]
     -output:
        dMatchesList: list of n DMatch object
     """
    dMatchesList = []
    for row in matchesList:
        dMatchesList.append(cv2.DMatch(_queryIdx=row[0], _trainIdx=row[1], _distance=row[2]))
    return dMatchesList

def indexMatrixToMatchesList2(indexMatrix):
    dMatchesList = []
    for i in range(len(indexMatrix)):
        if indexMatrix[i] != -1:
            dMatchesList.append(cv2.DMatch(_queryIdx=i, _trainIdx=indexMatrix[i], _distance=0))
    return dMatchesList

def matchesListToIndexMatrix(dMatchesList):
    """
     -input:
         dMatchesList: list of n DMatch object
     -output:
        matchesList: nMatches x 3 --> [[indexDesc1,indexDesc2,descriptorDistance],...]]
     """
    matchesList = []
    for k in range(len(dMatchesList)):
        matchesList.append([int(dMatchesList[k].queryIdx), int(dMatchesList[k].trainIdx), dMatchesList[k].distance])
    return matchesList

def matchWith2NDRR_2(desc1, desc2, distRatio, minDist):
    """
    Nearest Neighbours Matching algorithm checking the Distance Ratio.
    A match is accepted only if its distance is less than distRatio times
    the distance to the second match.
    -input:
        desc1: descriptors from image 1 nDesc x 128
        desc2: descriptors from image 2 nDesc x 128
        distRatio:
    -output:
       matches: nMatches x 3 --> [[indexDesc1,indexDesc2,descriptorDistance],...]]
    """
    matches = []
    nDesc1 = desc1.shape[0]
    for kDesc1 in range(nDesc1):
        dist = np.sqrt(np.sum((desc2 - desc1[kDesc1, :]) ** 2, axis=1))
        indexSort = np.argsort(dist)
        # print(dist[indexSort[0]], " ", dist[indexSort[1]])
        # Find the second closest match
        if (dist[indexSort[1]]*distRatio < dist[indexSort[0]]):
            matches.append([kDesc1, indexSort[0], dist[indexSort[0]]])
    return matches

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
    # print(points1.shape[0], " ", points1.shape[1])
    A = np.zeros((points1.shape[1], 9))
    for i in range(points1.shape[1]):
        A[i, :] = [points1[0, i] * points2[0, i], points2[0, i] * points1[1, i], points2[0, i], points1[0, i] * points2[1, i], points1[1, i] * points2[1, i], points2[1,i], points1[0,i], points1[1,i], 1]
        # A[i, :] = [points1[i,0] * points2[i,0], points2[i,0] * points1[i,0], points2[i,0], points1[i,0] * points2[i, 1], points1[i, 1] * points2[i, 1], points2[i,1], points1[i,0], points1[i,1], 1]
    
    _, _, V = np.linalg.svd(A)

    # compute the fundamental matrix from the right singular vector corresponding to the smallest singular value
    F = V[-1, :].reshape((3, 3))
    U, S, V = np.linalg.svd(F)

    # enforce rank 2 constraint
    S[2] = 0
    F = np.dot(U, np.dot(np.diag(S), V))
    F = np.dot(N2.T, np.dot(F, N1))
    return F/F[2,2]


def compute_fundamental_matrix(points1, points2, nx1, ny1, nx2, ny2):
    # Normalize the points
    # N1 = normalizationMatrix(nx1, ny1)
    # N2 = normalizationMatrix(nx2, ny2)
    # points1 = N1 @ points1
    # points2 = N2 @ points2
    # Compute the fundamental matrix
    # print(points1.shape[0], " ", points1.shape[1])
    A = np.zeros((points1.shape[1], 9))
    for i in range(points1.shape[1]):
        A[i, :] = [points1[0, i] * points2[0, i], points2[0, i] * points1[1, i], points2[0, i], points1[0, i] * points2[1, i], points1[1, i] * points2[1, i], points2[1,i], points1[0,i], points1[1,i], 1]
        # A[i, :] = [points1[i,0] * points2[i,0], points2[i,0] * points1[i,0], points2[i,0], points1[i,0] * points2[i, 1], points1[i, 1] * points2[i, 1], points2[i,1], points1[i,0], points1[i,1], 1]
    
    _, _, V = np.linalg.svd(A)

    # compute the fundamental matrix from the right singular vector corresponding to the smallest singular value
    F = V[-1, :].reshape((3, 3))
    U, S, V = np.linalg.svd(F)

    # enforce rank 2 constraint
    S[2] = 0
    F = np.dot(U, np.dot(np.diag(S), V))
    # F = np.dot(N2.T, np.dot(F, N1))
    return F/F[2,2]

# Triangulate a set of points given the projection matrices of two cameras.
def triangulation(P1, P2, points1, points2):    
    points3D = np.zeros((4, points1.shape[1]))
    for i in range(points1.shape[1]):
        p1 = points1[:, i].reshape(2, 1)
        p2 = points2[:, i].reshape(2, 1)
        A = [p1[0] * P1[2, :] - P1[0, :], p1[1] * P1[2, :] - P1[1, :], p2[0] * P2[2, :] - P2[0, :], p2[1] * P2[2, :] - P2[1, :]]
        _, _, V = np.linalg.svd(A)
        X = V[-1, :]
        points3D[:, i] = X / X[3]

    return points3D

def points_in_front_of_both_cameras(x1, x2, T, K):
    I = np.array([[1, 0 , 0, 0], [0, 1, 0, 0], [0, 0, 1 ,0]])

    P1 = K @ I
    P2 = K @ T

    points3d = triangulation(P1, P2, x1, x2)
    points3d = points3d.T
    # print(points3d.shape)

    in_front = 0

    points_front = []
    for point in points3d:
        points_front.append(point)
        if point[2] <= 0:
            continue

        # z > 0 in C1 frame
        pointFrame = T @ point.reshape(-1,1)
        if pointFrame[2] > 0:
            in_front += 1
        
    return np.array(in_front), np.array(points_front)

# Decompose the Essential matrix
def decompose_essential_matrix(x1, x2, E, K):
    # Compute the SVD of the essential matrix
    U, _, V = np.linalg.svd(E)
    t = U[:,2].reshape(-1,1)
    
    # Ensure that the determinant of U and Vt is positive (to ensure proper rotation)

    W = np.array([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1]
    ])

    R_90 = U @ W @ V if np.linalg.det(U @ W @ V) > 0 else -U @ W.T @ V
    R_n90 = U @ W.T @ V if np.linalg.det(U @ W.T @ V) > 0 else -U @ W.T @ V

    # Compute the four possible solutions
    solutions = []
    solutions.append(np.hstack((R_90,U[:,2].reshape(-1,1)))) #R + 90 + t
    solutions.append(np.hstack((R_90,-U[:,2].reshape(-1,1)))) # R + 90 - t
    solutions.append(np.hstack((R_n90,U[:,2].reshape(-1,1))))  # R - 90 + t
    solutions.append(np.hstack((R_n90,-U[:,2].reshape(-1,1)))) # R - 90 - t
    

    # points_front = [ points_in_front_of_both_cameras(x1, x2, T, K) for T in solutions]

    points_front = []
    points = []
    for T in solutions:
        v1, v2 = points_in_front_of_both_cameras(x1, x2, T, K)
        print(v1, v2.shape)
        points_front.append(v1)
        points.append(v2)
    # print(np.argmax(points_front))
    T = solutions[np.argmax(points_front)]
    return T, np.array(points[np.argmax(points_front)])

def compute_essential_matrix(F, K):
    E = K.T @ F @ K
    return E

def crossMatrixInv(M):
    x = [M[2, 1], M[0, 2], M[1, 0]]
    return x

def crossMatrix(x):
    M = np.array([[0, -x[2], x[1]],
    [x[2], 0, -x[0]],
    [-x[1], x[0], 0]], dtype="object")
    return M


def resBundleProjection(Op, x1Data, x2Data, K_c, nPoints):
    """
    -input:
    Op: Optimization parameters: this must include a
    paramtrization for T_21 (reference 1 seen from reference 2)
    in a proper way and for X1 (3D points in ref 1)
    x1Data: (3xnPoints) 2D points on image 1 (homogeneous
    coordinates)
    x2Data: (3xnPoints) 2D points on image 2 (homogeneous
    coordinates)
    K_c: (3x3) Intrinsic calibration matrix
    nPoints: Number of points
    -output:
    res: residuals from the error between the 2D matched points
    and the projected points from the 3D points
    (2 equations/residuals per 2D point)
    """

    '''
    Op[0:1] -> theta, phi
    Op[2:5] -> Rx,Ry,Rz
    Op[5:5 + nPoints*3] -> 3DXx,3DXy,3DXz
    '''
    # Bundle adjustment using least squares function
    idem = np.append(np.eye(3), np.zeros((3, 1)), axis=1)
    R = sc.linalg.expm(crossMatrix(Op[2:5]))
    t = np.array([np.sin(Op[0])*np.cos(Op[1]), np.sin(Op[0])*np.sin(Op[1]), np.cos(Op[0])]).reshape(-1,1)
    theta_ext_1 = K_c @ idem
    T = np.hstack((R, t))
    theta_ext_2 =  K_c @ T #Proyection matrix

    # print(Op[5:].shape, x1Data.shape, x2Data.shape, K_c.shape, nPoints)
    # Compute the 3D points
    X_3D = np.hstack((Op[5:].reshape(-1, 3), np.ones((nPoints, 1))))

    projection1 = theta_ext_1 @ X_3D.T
    projection1 = projection1[:2, :] / projection1[2, :]
    res1 = x1Data[:, :nPoints].T - projection1.T

    projection2 = theta_ext_2 @ X_3D.T
    projection2 = projection2[:2, :] / projection2[2, :]
    res2 = x2Data[:, :nPoints].T - projection2.T

    res = np.hstack((res1, res2)).flatten()

    return np.array(res)

def intersection(k1, k1_2, k2, k3):
    print(k1.shape, k1_2.shape, k2.shape, k3.shape)
    k2_n = []
    k3_n = []
    k1_n = []
    i = 0
    for k in k1:
        j = 0
        for k2_ in k1_2:
            if np.linalg.norm(np.array(k) - np.array(k2_)) <= 4:    
                k2_n.append(k2[i,:])
                k1_n.append(k)
                k3_n.append(k3[j,:])
            j += 1
        i += 1
    return np.array(k1_n), np.array(k2_n), np.array(k3_n)

"the unknowns are the camera matrix parameters"
"Each 2D-3D correspondence gives rise to two equations"
def DLTcamera(matches, x_3d):

    A = np.zeros((2*len(matches), 12))

    for i in range(len(matches)):
        A[2*i,:] = np.array([-x_3d[i,0], -x_3d[i,1], -x_3d[i,2], -x_3d[i,3], 0, 0, 0, 0, matches[i][0]*x_3d[i,0], matches[i][0]*x_3d[i,1], matches[i][0]*x_3d[i,2], matches[i][0] * x_3d[i,3]])
        A[2*i+1,:] = np.array([0, 0, 0, 0, -x_3d[i,0], -x_3d[i,1], -x_3d[i,2], -x_3d[i,3], matches[i][1]*x_3d[i,0], matches[i][1]*x_3d[i,1], matches[i][1]*x_3d[i,2], matches[i][1] * x_3d[i,3]])

    _, _, V = np.linalg.svd(A)
    P = V[-1,:].reshape((3,4))
    return P

def decompose_P_matrix(P):
    # Computing the optical centre pose in world frame solving svd(P)
    _, _, V = np.linalg.svd(P)
    C = V[-1, :]
    C = C / C[3]
    C = C[0:3]

    # rq decomposition of the camera matrix
    K, R = sc.linalg.rq(P[:, 0:3])

    # Check that the diagonal elements of K are positive
    D = np.diag(np.sign(np.diag(K)))
    R = D @ R
    K_raya = D @ K
    K = K_raya / K_raya[2, 2]

    # Compute the translation vector
    t = np.linalg.inv(K) @ P[:, 3]
    return R, t, K

def resBundleProjection_n_cameras(Op, xData, nCameras, K_c, nPoints):
    """
    -input:
    Op: Optimization parameters: this must include a
    paramtrization for T_21 (reference 1 seen from reference 2)
    in a proper way and for X1 (3D points in ref 1)
    x1Data: (3xnPoints) 2D points on image 1 (homogeneous
    coordinates)
    x2Data: (3xnPoints) 2D points on image 2 (homogeneous
    coordinates)
    K_c: (3x3) Intrinsic calibration matrix
    nPoints: Number of points
    -output:
    res: residuals from the error between the 2D matched points
    and the projected points from the 3D points
    (2 equations/residuals per 2D point)

    ASSUMING AT LEAST 3 CAMERAS !!!
    """

    '''
    Op[0:1] -> theta, phi
    Op[2:5] -> Rx,Ry,Rz
    Op[6:8] -> tx, ty, tz (camera 3 in advance)
    Op[9:11] -> Rx,Ry,Rz
    ...
    Op[] -> 3DXx,3DXy,3DXz
    '''
    # Bundle adjustment using least squares function
    idem = np.append(np.eye(3), np.zeros((3, 1)), axis=1)

    theta_ext_1 = K_c @ idem


    theta_ext = []
    theta_ext.append(theta_ext_1)

    for i in range(nCameras - 1):
        # R = sc.linalg.expm(crossMatrix(Op[2+5*i:5+5*i]))
        if i == 0:
            # No se que hay que poner aqui
            t = np.array([np.sin(Op[0])*np.cos(Op[1]), np.sin(Op[0])*np.sin(Op[1]), np.cos(Op[0])]).reshape(-1,1)
            R = sc.linalg.expm(crossMatrix(Op[2+5*i:5+5*i]))
        else:
            t = np.array([Op[6*(i-1)+5],Op[6*(i-1)+6],Op[6*(i-1)+7]]).reshape(-1,1)
            R = sc.linalg.expm(crossMatrix(Op[6*(i-1)+8:6*(i-1)+11]))
            if i == 2:
                K_c = np.array(Op[6*(i-1)+11:6*(i-1)+20]).reshape(3,3)
        T = np.hstack((R, t))
        theta_ext.append(K_c @ T)


    # Compute the residuals
   
    Xpoints = xData
    # Xpoints = xData.reshape(nCameras, nPoints, 2)
    idx_3D = 5 + (nCameras-2)*6 + 9
    X_3D = np.hstack((Op[idx_3D:].reshape(-1, 3), np.ones((nPoints, 1))))
    # print(X_3D)
    res = []
    for i in range(nCameras):
        projection = theta_ext[i] @ X_3D.T
        projection = projection[:2, :] / projection[2, :]
        res.append((Xpoints[i] - projection.T).flatten())

    # print(res)
    return np.array(res).flatten()

def ensamble_T(R_w_c, t_w_c) -> np.array:
    """
    Ensamble the a SE(3) matrix with the rotation matrix and translation vector.
    """
    T_w_c = np.zeros((4, 4))
    T_w_c[0:3, 0:3] = R_w_c
    T_w_c[0:3, 3] = t_w_c
    T_w_c[3, 3] = 1
    return T_w_c

def calculate_RANSAC_own_F(source,dst,third, threshold, nx1, ny1, nx2, ny2):
    num_samples = 8
    spFrac = 0.6  # spurious fraction
    P = 0.999  # probability of selecting at least one sample without spurious

    # number m of random samples
    nAttempts = np.round(np.log(1 - P) / np.log(1 - np.power((1 - spFrac),num_samples)))
    num_attempts = nAttempts.astype(int)
    num_attempts = 10000

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
                best_model_matches = matches_subset
                best_model_matches = np.hstack((best_model_matches, np.array(good_matches).T))

    return F_most_voted, best_model_matches


if __name__ == '__main__':
    Kc_new = np.loadtxt('calibration_matrix.txt')


    path_image_1 = 'imgs1/img_new1.jpg'
    path_image_2 = 'imgs1/img_new2.jpg'
    path_image_old = 'imgs1/img_old.jpg'

    img1 = cv2.imread(path_image_1)
    img2 = cv2.imread(path_image_2)
    img_old = cv2.imread(path_image_old)

    print('Imgs loaded')

    path = './output/img_new1_img_new2_matches.npz'
    npz = np.load(path)
    keypoints_SG_0 = npz['keypoints0']
    keypoints_SG_1 = npz['keypoints1']
    path2 = './output/img_new1_img_old_matches.npz'
    npz2 = np.load(path2)
    keypoints_SG_0_old = npz2['keypoints0']
    keypoints_SG_1_old = npz2['keypoints1']

    matchesListSG_0 = [i for i, x in enumerate(npz['matches']) if x != -1]
    matchesListSG_1 = [x for i, x in enumerate(npz['matches']) if x != -1]
    matchesListSG_0_old = [i for i, x in enumerate(npz2['matches']) if x != -1]
    matchesListSG_1_old = [x for i, x in enumerate(npz2['matches']) if x != -1]

    

    # Matched points from SuperGlue
    srcPts_SG = np.float32([keypoints_SG_0[m] for m in matchesListSG_0])
    dstPts_SG = np.float32([keypoints_SG_1[m] for m in matchesListSG_1])
    srcPts_SG_old = np.float32([keypoints_SG_0[m] for m in matchesListSG_0])
    dstPts_SG_old = np.float32([keypoints_SG_1[m] for m in matchesListSG_1])

    keypoints_SG_0, keypoints_SG_1, keypoints_SG_1_old = intersection(srcPts_SG, srcPts_SG_old, dstPts_SG, dstPts_SG_old)

    print(keypoints_SG_0.shape)
    print(keypoints_SG_1.shape)
    print(keypoints_SG_1_old.shape)

    srcPts_SG = keypoints_SG_0
    dstPts_SG = keypoints_SG_1
    dstPts_SG_old = keypoints_SG_1_old

    
    F, good_matches = calculate_RANSAC_own_F(srcPts_SG.T, dstPts_SG.T, dstPts_SG_old.T, 2, img1.shape[1], img1.shape[0], img2.shape[1], img2.shape[0])
    print(F)
    print(good_matches.shape)

    # Matched points in numpy from list of DMatches
    srcPts = np.float32([keypoints_SG_0[m.queryIdx] for m in dMatchesList]).reshape(len(dMatchesList), 2)
    dstPts = np.float32([keypoints_SG_1[m.trainIdx] for m in dMatchesList]).reshape(len(dMatchesList), 2)
    dstPts_old = np.float32([keypoints_SG_1_old[m.trainIdx] for m in dMatchesList]).reshape(len(dMatchesList), 2)
    print(srcPts.shape, dstPts.shape)

    # Fundamental matrix from SuperGlue
    print(srcPts.shape, dstPts.shape, dstPts_old.shape)
    F_SG = compute_fundamental_matrix(srcPts.T, dstPts.T, img1.shape[1], img1.shape[0], img2.shape[1], img2.shape[0])

    print('Fundamental matrix from SuperGlue: \n', F_SG)

    E_SG = compute_essential_matrix(F_SG, Kc_new)
    print('Essential matrix from SuperGlue: \n', E_SG)
    Rt, X_3d = decompose_essential_matrix(srcPts.T, dstPts.T, E_SG, Kc_new)
    print(X_3d.shape)
    print('Rt from SuperGlue: \n', Rt)
    # # add last row to make it 4x4
    R_c2_c1 = Rt[:, 0:3]
    t_c2_c1 = Rt[:, 3].reshape(-1,1)

    elevation = np.arccos(t_c2_c1[2] / np.linalg.norm(t_c2_c1))
    azimuth = np.arctan2(t_c2_c1[1], t_c2_c1[0])

    Op = [elevation, azimuth] + crossMatrixInv(sc.linalg.logm(R_c2_c1)) + X_3d[:,0:3].flatten().tolist()

    # Bundle adjustment using least squares function
    # OpOptim = scOptim.least_squares(resBundleProjection, Op, args=(srcPts.T, dstPts.T, Kc_new, srcPts.shape[0]),method='trf', loss='huber', verbose=2, ftol=1e-4, xtol=1e-4, gtol=1e-4, max_nfev=1000)
    # np.savetxt('Optimization.txt', OpOptim.x)

    # OpOptim = OpOptim.x

    OpOptim = np.loadtxt('Optimization.txt')

    R_c2_c1 = sc.linalg.expm(crossMatrix(OpOptim[2:5]))
    t_c2_c1 = np.array([np.sin(OpOptim[0])*np.cos(OpOptim[1]), np.sin(OpOptim[0])*np.sin(OpOptim[1]), np.cos(OpOptim[0])]).reshape(-1,1)
    
    T_c2_c1_op = ensamble_T(R_c2_c1, t_c2_c1.T)

    points_3D_Op = np.concatenate((OpOptim[5: 8], np.array([1.0])), axis=0)

    for i in range(X_3d.shape[0]-1):
        points_3D_Op = np.vstack((points_3D_Op, np.concatenate((OpOptim[8+3*i: 8+3*i+3], np.array([1.0])) ,axis=0)))

    

    idem = np.append(np.eye(3), np.zeros((3, 1)), axis=1)
    # P1_est = Kc_new @ idem
    # x1_p = P1_est @ points_3D_Op.T
    # x1_p = x1_p / x1_p[2, :]
    # x2_p = P2_op @ points_3D_Op.T
    # x2_p = x2_p / x2_p[2, :]

    points_Op = idem @ (points_3D_Op).T

    # fig3D = plt.figure(2)
    # ax = plt.axes(projection='3d', adjustable='box')
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')

    # drawRefSystem(ax, np.eye(4,4), '-', 'C1')
    # drawRefSystem(ax, np.eye(4,4) @ np.linalg.inv(T_c2_c1_op), '-', 'C2_BA_scaled')

    # ax.scatter(points_Op[0, :], points_Op[1, :], points_Op[2, :], marker='.')  # 3D points

    # plt.title('3D points Bundle adjustment')
    # plt.show()

    # fig3D = plt.figure(2)
    # ax = plt.axes(projection='3d', adjustable='box')
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # ax.scatter(points_Op[0, :], points_Op[1, :], points_Op[2, :], marker='.')  # 3D points

    # plt.title('3D points Bundle adjustment')
    # plt.show()

    points_C3 = dstPts_old

    print(points_3D_Op.shape, points_C3.shape)

    P_old = DLTcamera(points_C3, points_3D_Op)
    print(P_old)
    R_c3_c1, t_c3_c1, K_c3_c1 = decompose_P_matrix(P_old)
    print(R_c3_c1)
    print(t_c3_c1)
    print(K_c3_c1)
    T_c3_c1 = ensamble_T(R_c3_c1, t_c3_c1)
    print(T_c3_c1)
   
    # DLT camera matrix P


    fig3D = plt.figure(8)

    ax = plt.axes(projection='3d', adjustable='box')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    drawRefSystem(ax, np.eye(4, 4), '-', 'C1')
    drawRefSystem(ax, np.eye(4,4) @ np.linalg.inv(T_c3_c1), '-', 'C3_PnP')
    drawRefSystem(ax, np.eye(4,4) @ np.linalg.inv(T_c2_c1_op), '-', 'C2_BA_scaled')
    plt.title('3D camera poses PnP')
    plt.draw()
    plt.show()

    elevation2 = np.arccos(t_c2_c1[2])
    azimuth2 = np.arcsin(t_c2_c1[0] / np.sin(elevation2))

    Kc_old = K_c3_c1.flatten().tolist()
    print(Kc_old)

    Op2 = [elevation2, azimuth2] + crossMatrixInv(sc.linalg.logm(R_c2_c1)) + [t_c3_c1[0], t_c3_c1[1], t_c3_c1[2]] + crossMatrixInv(sc.linalg.logm(R_c3_c1)) + K_c3_c1.flatten().tolist() + X_3d[:,0:3].flatten().tolist()

    X2 = np.stack((srcPts, dstPts, points_C3))
    OpOptim2 = scOptim.least_squares(resBundleProjection_n_cameras, Op2, args=(X2, 3, Kc_new, srcPts.shape[0]),method='trf', loss='huber', verbose=2, ftol=1e-5, xtol=1e-5, gtol=1e-5, max_nfev=1000)
    np.savetxt('Optimization2.txt', OpOptim2.x)

    OpOptim2 = OpOptim2.x

    R_c2_c1 = sc.linalg.expm(crossMatrix(OpOptim2[2:5]))
    t_c2_c1 = np.array([np.sin(OpOptim2[0])*np.cos(OpOptim2[1]), np.sin(OpOptim2[0])*np.sin(OpOptim2[1]), np.cos(OpOptim2[0])]).reshape(-1,1)
    T_c2_c1_op = ensamble_T(R_c2_c1, t_c2_c1.T)

    R_c3_c1 = sc.linalg.expm(crossMatrix(OpOptim2[8:11]))
    t_c3_c1 = np.array([OpOptim2[5],OpOptim2[6],OpOptim2[7]]).reshape(-1,1)
    T_c3_c1_op = ensamble_T(R_c3_c1, t_c3_c1.T)
    Kc_old = np.array(OpOptim2[11:20]).reshape(3,3)

    points_3D_Op = np.concatenate((OpOptim2.x[20: 20+3], np.array([1.0])), axis=0)

    for i in range(X_3d.shape[0]-1):
        points_3D_Op = np.vstack((points_3D_Op, np.concatenate((OpOptim2.x[20+3+3*i: 20+3+3*i+3], np.array([1.0])) ,axis=0)))


    #### Draw 3D ################
    fig3D = plt.figure(2)
    ax = plt.axes(projection='3d', adjustable='box')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    drawRefSystem(ax, np.eye(4, 4), '-', 'C1')
    drawRefSystem(ax, np.eye(4,4) @ np.linalg.inv(T_c2_c1_op), '-', 'C2_BA_scaled')
    drawRefSystem(ax, np.eye(4,4) @ np.linalg.inv(T_c3_c1_op), '-', 'C3_BA_scaled')
    
