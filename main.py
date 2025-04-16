import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # nécessaire pour projection='3d'

def fondamental_matrix(l_x1, l_x2):
    '''
    Entrée : 
        l_x1 : liste de points de l'image 1 (list of [x, y])
        l_x2 : liste de points correspondants de l'image 2 (list of [x, y])
        l_x1[i] correspond à l_x2[i] et len(l_x1) = len(l_x2)
    Sortie : 
        La matrice fondamentale F (3x3)
    Algorithme : Résoud les équations matricielles (l_x1[i])T * F * (l_x2[i]) = 0 pour tout i 
    '''
    assert len(l_x1) == len(l_x2) and len(l_x1) >= 8, "Il faut au moins 8 correspondances"
    # Convertir les points en format correct (float32)
    l_x1 = np.array(l_x1, dtype=np.float32)
    l_x2 = np.array(l_x2, dtype=np.float32)
    
    # Trouver la matrice fondamentale en utilisant RANSAC pour une estimation robuste
    F, mask = cv2.findFundamentalMat(l_x1, l_x2, cv2.FM_RANSAC)
    
    return F


# Calcul de P et P'
'''def normalize_points(points):
    mean = np.mean(points, axis=0)
    std = np.std(points, axis=0)
    T = np.array([
        [1/std[0], 0, -mean[0]/std[0]],
        [0, 1/std[1], -mean[1]/std[1]],
        [0, 0, 1]
    ])
    return (T @ points.T).T, T'''

def compute_epipole(F):
    _, _, Vt = np.linalg.svd(F)
    e_prime = Vt[-1, :]
    e_prime /= e_prime[2] if not np.isclose(e_prime[2], 0) else 1e-10
    return e_prime

def compute_P_prime(F, e_prime):
    e_prime_skew = np.array([
        [0, -e_prime[2], e_prime[1]],
        [e_prime[2], 0, -e_prime[0]],
        [-e_prime[1], e_prime[0], 0]
    ])
    return np.hstack((e_prime_skew @ F, e_prime.reshape(3, 1)))





# triangulation
def triangulate_points_dlt_all(l_x1, l_x2, P, P_prime):
    """
    Triangule tous les points correspondants entre deux images
    avec la méthode DLT (Direct Linear Transform).

    Entrée :
        l_x1 : liste de N points (N x 2) dans image 1
        l_x2 : liste de N points (N x 2) dans image 2
        P, P_prime : matrices de projection (3x4) des caméras

    Sortie :
        points_3d : tableau (N x 3) contenant les points 3D reconstruits
    """
    
    points_3d = []  # Liste pour stocker les points 3D

    # On parcourt chaque paire de points correspondants
    for i in range(len(l_x1)):
        u, v = l_x1[i]       # coordonnées dans image 1
        u_p, v_p = l_x2[i]   # coordonnées dans image 2

        # Construction de la matrice A (4x4) à partir des équations projetées :
        # x = P X  =>  u * P3 - P1,  v * P3 - P2
        # x' = P' X =>  u' * P'3 - P'1, v' * P'3 - P'2
        A = np.array([
            u * P[2] - P[0],              # Première ligne de A : u * P3 - P1
            v * P[2] - P[1],              # Deuxième ligne : v * P3 - P2
            u_p * P_prime[2] - P_prime[0],# Troisième ligne : u' * P'3 - P'1
            v_p * P_prime[2] - P_prime[1] # Quatrième ligne : v' * P'3 - P'2
        ])

        # Décomposition en valeurs singulières : A = U Σ V^T
        _, _, Vt = np.linalg.svd(A)

        # Le vecteur X recherché (en coordonnées homogènes) est la dernière ligne de V^T
        X = Vt[-1]

        # On divise par X[3] pour passer des coordonnées homogènes à euclidiennes
        X = X / X[-1]

        # On ne garde que les coordonnées 3D (X, Y, Z)
        points_3d.append(X[:3])

    # On retourne tous les points sous forme de tableau (N x 3)
    return np.array(points_3d)




# ---------------------
# Exemple principal
# ---------------------
# Exemple de points 2D dans l'image 1
l_x1 = np.array([
    [150, 200],  # Point 1
    [320, 180],  # Point 2
    [500, 250],  # Point 3
    [600, 300],  # Point 4
    [180, 220],  # Point 5
    [330, 240],  # Point 6
    [480, 270],  # Point 7
    [550, 350]   # Point 8
])

# Exemple de points correspondants dans l'image 2 (décalés)
l_x2 = np.array([
    [152, 202],  # Point 1 (décalé)
    [322, 182],  # Point 2 (décalé)
    [502, 252],  # Point 3 (décalé)
    [602, 302],  # Point 4 (décalé)
    [182, 222],  # Point 5 (décalé)
    [332, 242],  # Point 6 (décalé)
    [482, 272],  # Point 7 (décalé)
    [552, 352]   # Point 8 (décalé)
])

# 1. Calcul de la matrice fondamentale
F = fondamental_matrix(l_x1, l_x2)
print("Matrice Fondamentale F:\n", F)

# 2. Matrices de projection
P = np.hstack((np.eye(3), np.zeros((3, 1))))         # Camera 1 à l'origine
e_prime = compute_epipole(F)                         # Épipôle pour la caméra 2
P_prime = compute_P_prime(F, e_prime)                # Matrice de projection de la caméra 2
print("\nMatrice de Projection P':\n", P_prime)

# 3. Triangulation d'un point (avec OpenCV)
point_3d = cv2.triangulatePoints(P, P_prime, l_x1[0:1].T, l_x2[0:1].T)
#point_3d /= point_3d[3]
print("\nPoint 3D (OpenCV) reconstruit:", point_3d[:3].T)

# 4. Triangulation de tous les points (DLT)
points_3d = triangulate_points_dlt_all(l_x1, l_x2, P, P_prime)
print("\nTous les points 3D (DLT) reconstruits :\n", points_3d)

# ---------- Visualisation 3D ----------
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Points 3D en rouge
ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], c='red', label='Points 3D')

ax.set_title("Visualisation des points 3D triangulés (DLT)")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.legend()
plt.show()