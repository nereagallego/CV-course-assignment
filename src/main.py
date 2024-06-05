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

    Kc = np.loadtxt('calibration_matrix.txt')

    # Load images
    img1 = cv2.imread('imgs/img_new1.jpg')
    img2 = cv2.imread('imgs/img_new2.jpg')
    img3 = cv2.imread('imgs/img_new3.jpg')
    img4 = cv2.imread('imgs/img_new4.jpg')
    img_old = cv2.imread('imgs/img_old.jpg')