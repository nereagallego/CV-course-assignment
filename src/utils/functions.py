import numpy as np
import scipy as sc

def ensamble_T(R_w_c, t_w_c) -> np.array:
    """
    Ensamble the a SE(3) matrix with the rotation matrix and translation vector.
    """
    T_w_c = np.zeros((4, 4))
    T_w_c[0:3, 0:3] = R_w_c
    T_w_c[0:3, 3] = t_w_c
    T_w_c[3, 3] = 1
    return T_w_c

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

    for _ in range(num_attempts):
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

# Triangulate a set of points given the projection matrices of two cameras.
def triangulation(P1, P2, points1, points2):    
    points3D = np.zeros((4, points1.shape[1]))
    for i in range(points1.shape[1]):
        p1 = points1[:, i]
        p2 = points2[:, i]
        A = np.vstack((p1[0] * P1[2, :] - P1[0, :], 
                       p1[1] * P1[2, :] - P1[1, :], 
                       p2[0] * P2[2, :] - P2[0, :], 
                       p2[1] * P2[2, :] - P2[1, :]))
        _, _, V = np.linalg.svd(A)
        X = V[-1, :]
        X = X / X[3]  # Normalize the 3D point
        points3D[:, i] = X

    return points3D  

# Estimate the camera pose from the Essential matrix
def sfm(F, K_c, x1, x2, x3):
    # Compute the essential matrix
    E = K_c.T @ F @ K_c

    # Compute SVD of the essential matrix
    U, _, V = np.linalg.svd(E)

    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

    # Compute the four possible camera poses
    R1 = U @ W @ V
    R2 = U @ W.T @ V

    t1 = U[:, 2]
    t2 = -U[:, 2]

    # Compute the four possible projection matrices
    P1 = K_c @ np.hstack((np.eye(3), np.zeros((3, 1))))

    Ts = []
    for R, t in zip([R1, R1, R2, R2], [t1, t2, t1, t2]):
        if np.linalg.det(R) < 0:
            R *= -1
            t *= -1
        Ts.append(np.hstack((R, t.reshape(3, 1))))

    # Triangulate the points, check if the triangulation is in front of the cameras and select the best solution
    X = []
    max = 0
    best = None
    for T in Ts:
        P = K_c @ T
        x_3d = triangulation(P1, P, x1, x2)
        points_front = []
        for i in range(x_3d.shape[1]):
            if x_3d[2,i] > 0  and np.dot(T[2, 0:3], x_3d[:3,i] - T[0:3, 3]) > 0:
                points_front.append(x_3d[:,i])
        if len(points_front) > max:
            max = len(points_front)
            best = T
            X = x_3d.T
            
    
    if best is None:
        print("No solution found")
        return None, None
    
    return ensamble_T(best[0:3, 0:3], best[0:3, 3]), X


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

"the unknowns are the camera matrix parameters"
"Each 2D-3D correspondence gives rise to two equations"
def DLTcamera(matches, x_3d):
    A = np.empty((0, 12))                               # DLT, diapo 13 tema 6
    for i in range(x_3d.shape[0]):   
        A = np.vstack((
            A,
            np.concatenate((-x_3d[i,:], np.zeros((4)), matches[i,0]*x_3d[i,:])),
            np.concatenate((np.zeros((4)), -x_3d[i,:], matches[i,1]*x_3d[i,:])),
        ))
    u, s, vh = np.linalg.svd(A)
    P3 = np.reshape(vh[-1, :], (3,4))
    return P3
    # A = np.zeros((2*len(matches), 12))

    # for i in range(len(matches)):
    #     A[2*i,:] = np.array([-x_3d[i,0], -x_3d[i,1], -x_3d[i,2], -x_3d[i,3], 0, 0, 0, 0, matches[i][0]*x_3d[i,0], matches[i][0]*x_3d[i,1], matches[i][0]*x_3d[i,2], matches[i][0] * x_3d[i,3]])
    #     A[2*i+1,:] = np.array([0, 0, 0, 0, -x_3d[i,0], -x_3d[i,1], -x_3d[i,2], -x_3d[i,3], matches[i][1]*x_3d[i,0], matches[i][1]*x_3d[i,1], matches[i][1]*x_3d[i,2], matches[i][1] * x_3d[i,3]])

    # _, _, V = np.linalg.svd(A)
    # P = V[-1,:].reshape((3,4))

    # # Normalize P
    # P /= P[2,3]

    # return P
    

