# Computer Vision Course Assignment
# Author: Nerea Gallego (801950)

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

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
        # Find the second closest match
        if (dist[indexSort[1]]*distRatio < dist[indexSort[0]]):
            matches.append([kDesc1, indexSort[0], dist[indexSort[0]]])
    return matches

if __name__ == '__main__':
    Kc_new = np.loadtxt('calibration_matrix.txt')


    path_image_1 = 'imgs1/img_new1.jpg'
    path_image_2 = 'imgs1/img_new2.jpg'

    img1 = cv2.imread(path_image_1)
    img2 = cv2.imread(path_image_2)

    print('Imgs loaded')

    path = './output/img_new1_img_new2_matches.npz'
    npz = np.load(path)
    keypoints_SG_0 = npz['keypoints0']
    keypoints_SG_1 = npz['keypoints1']
    matchesListSG_0 = [i for i, x in enumerate(npz['matches']) if x != -1]
    matchesListSG_1 = [x for i, x in enumerate(npz['matches']) if x != -1]

    # Matched points from SuperGlue
    srcPts_SG = np.float32([keypoints_SG_0[m] for m in matchesListSG_0])
    dstPts_SG = np.float32([keypoints_SG_1[m] for m in matchesListSG_1])
    
    # Matched points in homogeneous coordinates
    x1_SG = np.vstack((srcPts_SG.T, np.ones((1, srcPts_SG.shape[0]))))
    x2_SG = np.vstack((dstPts_SG.T, np.ones((1, dstPts_SG.shape[0]))))

