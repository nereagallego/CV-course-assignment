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
    F_most_voted = None

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
    t = U[:3, -1]
    
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
    
    if idx is None:
        T = solutions[np.argmax(points_front)]
        return T, np.array(points[np.argmax(points_front)])
    else:
        return solutions[idx], np.array(points[idx])

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

def plotResidual2(x,xProjected,strStyle, ax):
    """
        Plot the residual between an image point and an estimation based on a projection model.
         -input:
             x: Image points.
             xProjected: Projected points.
             strStyle: Line style.
         -output: None
         """

    # Plot the line between each point and its projection also plot the point with a blue dot and the projection with a red cross
    for k in range(x.shape[0]):
        ax.plot([x[k, 0], xProjected[k, 0]], [x[k, 1], xProjected[k, 1]], strStyle)
        ax.plot(x[k, 0], x[k, 1], 'bo')
        ax.plot(xProjected[k, 0], xProjected[k, 1], 'rx')

"the unknowns are the camera matrix parameters"
"Each 2D-3D correspondence gives rise to two equations"
def DLTcamera(matches, x_3d):

    A = np.zeros((2*len(matches), 12))

    for i in range(len(matches)):
        A[2*i,:] = np.array([-x_3d[i,0], -x_3d[i,1], -x_3d[i,2], -x_3d[i,3], 0, 0, 0, 0, matches[i][0]*x_3d[i,0], matches[i][0]*x_3d[i,1], matches[i][0]*x_3d[i,2], matches[i][0] * x_3d[i,3]])
        A[2*i+1,:] = np.array([0, 0, 0, 0, -x_3d[i,0], -x_3d[i,1], -x_3d[i,2], -x_3d[i,3], matches[i][1]*x_3d[i,0], matches[i][1]*x_3d[i,1], matches[i][1]*x_3d[i,2], matches[i][1] * x_3d[i,3]])

    _, _, V = np.linalg.svd(A)
    P = V[-1,:].reshape((3,4))

    # Normalize P
    P /= P[2,3]

    return P

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

    print(keypoints_new_1.shape, keypoints_new_2.shape, keypoints_old.shape)

    F, matches = calculate_RANSAC_own_F(keypoints_new_1.T, keypoints_new_2.T, keypoints_old.T, 2, img1.shape[1], img1.shape[0], img2.shape[1], img2.shape[0])

    kp_new1 = matches[0:3,:].T
    kp_new2 = matches[3:6,:].T
    kp_old = matches[6:9,:].T

    print(kp_new1.shape, kp_new2.shape, kp_old.shape)

    E = compute_essential_matrix(F, Kc_new)

    T_c2_c1, X_3d = decompose_essential_matrix(kp_new1[:,0:2].T, kp_new2[:,0:2].T, E, Kc_new)
    print(X_3d.shape)

    T_w_c1 = ensamble_T(np.diag((1, 1, 1)), np.zeros((3)))

    # Plot the 3D points
    fig = plt.figure()
    ax = plt.axes(projection='3d', adjustable='box')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    drawRefSystem(ax, T_w_c1, '-', 'C1')
    drawRefSystem(ax, T_w_c1 @ np.linalg.inv(T_c2_c1) , '-', 'C2')
    ax.scatter(X_3d[:,0], X_3d[:,1], X_3d[:,2], marker='.')
    xFakeBoundingBox = np.linspace(0, 4, 2)
    yFakeBoundingBox = np.linspace(0, 4, 2)
    zFakeBoundingBox = np.linspace(0, 4, 2)
    ax.plot(xFakeBoundingBox, yFakeBoundingBox, zFakeBoundingBox, 'w.')
    plt.show()

    zeros = np.zeros(3).reshape(3,1)
    idem = np.hstack((np.identity(3),zeros))
    aux_matrix = np.dot(Kc_new,idem)

    P1 = aux_matrix @ T_w_c1
    P2 = aux_matrix @ (T_w_c1 @ T_c2_c1)

    x1_p = P1 @ X_3d.T
    x1_p = x1_p / x1_p[2, :]
    x2_p = P2 @ X_3d.T
    x2_p = x2_p / x2_p[2, :]

    fig, ax = plt.subplots(1,2, figsize=(10,5))
    ax[0].imshow(img1)
    ax[0].set_title('Residuals after Bundle adjustment Image1')
    plotResidual2(kp_new1, x1_p.T, 'k-', ax[0])
    ax[1].imshow(img2)
    ax[1].set_title('Residuals after Bundle adjustment Image2')
    plotResidual2(kp_new2, x2_p.T, 'k-', ax[1])
    plt.show()

    R = T_c2_c1[0:3, 0:3]
    t = T_c2_c1[0:3, 3].reshape(-1,1)

    elevation = np.arccos(t[2])
    azimuth = np.arctan2(t[1], t[0])
    # azimuth = np.arccos(t[0] / np.arcsin(elevation))

    Op = [elevation, azimuth] + crossMatrixInv(sc.linalg.logm(R)) + X_3d[:,0:3].flatten().tolist()

    OpOptim = scOptim.least_squares(resBundleProjection, Op, args=(kp_new1[:,0:2].T, kp_new2[:,0:2].T, Kc_new, kp_new1.shape[0]), method='trf', loss='huber', verbose=2)

    R_c2_c1 = sc.linalg.expm(crossMatrix(OpOptim.x[2:5]))
    t_c2_c1 = np.array([np.sin(OpOptim.x[0])*np.cos(OpOptim.x[1]), np.sin(OpOptim.x[0])*np.sin(OpOptim.x[1]), np.cos(OpOptim.x[0])])
    T_c2_c1 = ensamble_T(R_c2_c1, t_c2_c1)
    points_3d = np.concatenate((OpOptim.x[5:8], np.array([1.0])), axis=0)
    for i in range(X_3d.shape[0]-1):
        points_3d = np.vstack((points_3d, np.concatenate((OpOptim.x[8+3*i: 8+3*i+3], np.array([1.0])) ,axis=0)))

    # Plot the 3D points
    fig = plt.figure()
    ax = plt.axes(projection='3d', adjustable='box')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    drawRefSystem(ax, T_w_c1, '-', 'C1')
    drawRefSystem(ax, T_w_c1 @ np.linalg.inv(T_c2_c1) , '-', 'C2')
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
    # R_c3_c1, t_c3_c1, K_c3_c1 = decompose_P_matrix(P_old)
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
    xFakeBoundingBox = np.linspace(0, 4, 2)
    yFakeBoundingBox = np.linspace(0, 4, 2)
    zFakeBoundingBox = np.linspace(0, 4, 2)
    ax.plot(xFakeBoundingBox, yFakeBoundingBox, zFakeBoundingBox, 'w.')
    plt.show()