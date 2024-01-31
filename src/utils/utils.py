import numpy as np

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