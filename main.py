import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import datasets
import multi_image_point_matcher

def compute_fundamental_matrix(l_x1, l_x2):
    l_x1 = np.array(l_x1, dtype=np.float32).reshape(-1, 1, 2)
    l_x2 = np.array(l_x2, dtype=np.float32).reshape(-1, 1, 2)
    F, mask = cv2.findFundamentalMat(l_x1, l_x2, cv2.FM_RANSAC)
    return F

def get_intrinsic_matrix_with_specs(image_shape):
    h, w = image_shape
    sensor_width_mm = 7.01
    focal_length_mm = 4.2
    fx = (focal_length_mm / sensor_width_mm) * w
    fy = fx
    cx, cy = w / 2, h / 2
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

def estimate_essential_matrix(F, K):
    return K.T @ F @ K

def decompose_essential_matrix_all(E):
    U, _, Vt = np.linalg.svd(E)
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    R1, R2 = U @ W @ Vt, U @ W.T @ Vt
    t = U[:, 2]
    return [(R1, t), (R1, -t), (R2, t), (R2, -t)]

def compute_projection_matrices(K, R, t):
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = K @ np.hstack((R, t.reshape(3, 1)))
    return P1, P2

def triangulate_point(x1, x2, P1, P2):
    A = np.array([
        x1[0] * P1[2] - P1[0],
        x1[1] * P1[2] - P1[1],
        x2[0] * P2[2] - P2[0],
        x2[1] * P2[2] - P2[1]
    ])
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    return X[:3] / X[3]

def choose_best_RT(candidates, l_x1, l_x2, K):
    max_positive_depth = 0
    best_P2 = None
    best_points = None
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    for R, t in candidates:
        P2 = K @ np.hstack((R, t.reshape(3, 1)))
        positive_depth = 0
        points_3D = []
        for i in range(len(l_x1)):
            X = triangulate_point(l_x1[i], l_x2[i], P1, P2)
            if X[2] > 0 and (R @ X + t)[2] > 0:
                positive_depth += 1
            points_3D.append(X)
        if positive_depth > max_positive_depth:
            max_positive_depth = positive_depth
            best_P2 = P2
            best_points = points_3D
    return P1, best_P2, np.array(best_points)

# --- MAIN WORKFLOW ---
#l_x1, l_x2 = multi_image_point_matcher.obtain_correspondances()
l_x1, l_x2 = datasets.generate_dataset(500)
image_shape = (1000, 1500)

K = get_intrinsic_matrix_with_specs(image_shape)
F = compute_fundamental_matrix(l_x1, l_x2)
E = estimate_essential_matrix(F, K)
candidates = decompose_essential_matrix_all(E)

P1, P2, points_3D = choose_best_RT(candidates, l_x1, l_x2, K)

# Visualisation
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points_3D[:, 0], points_3D[:, 1], points_3D[:, 2], c='red', marker='o')
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Points 3D reconstruits (meilleure décomposition)")
plt.show()
