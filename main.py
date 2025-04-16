import cv2
import numpy as np


'''def compute_fundamental_matrix(l_x1, l_x2):
    """
    Calcule la matrice fondamentale F à partir des points correspondants.
    pts1 et pts2 sont des listes de points correspondants dans les deux images.
    Ici à la différence de la méthode 1 on calcule F à l'aide d'une fonction prédéfinie
    """
    # Conversion des points en format requis par OpenCV (float32 et reshape)
    l_x1 = np.array(l_x1, dtype=np.float32).reshape(-1, 1, 2)
    l_x2 = np.array(l_x2, dtype=np.float32).reshape(-1, 1, 2)

    # Calcul de la matrice fondamentale en utilisant la méthode RANSAC pour améliorer la robustesse
    F, mask = cv2.findFundamentalMat(l_x1, l_x2, cv2.FM_RANSAC)
    #mask est un tableau de valeurs binaires (0 ou 1) qui indique si chaque paire de points correspondants a été considérée comme valide ou non lors de l'estimation de F

    return F, mask'''
    
def compute_fundamental_matrix(l_x1, l_x2):
    '''
    C celle d'Axel la fonction prédéfinie veut pas marcher 
    Entrée : 
        l_x1 : liste de points de l'image 1 (list of [x, y])
        l_x2 : liste de points correspondants de l'image 2 (list of [x, y])
        l_x1[i] correspond à l_x2[i] et len(l_x1) = len(l_x2)
    Sortie : 
        La matrice fondamentale F (3x3)
    Algorithme : Résoud les équations matricielles (l_x1[i])T * F * (l_x2[i]) = 0 pour tout i 
    '''
    assert len(l_x1) == len(l_x2) and len(l_x1) >= 8, "Il faut au moins 8 correspondances"

    # Mise en forme homogène des points
    x1 = np.array([ [x, y, 1] for x, y in l_x1 ])
    x2 = np.array([ [x, y, 1] for x, y in l_x2 ])

    # Construction de la matrice A
    A = [] # On trouve les valeurs de A en regardant ce que vaut X1^T * F * X2
    for i in range(len(x1)):
        X1 = x1[i]
        X2 = x2[i]
        A.append([
            X2[0]*X1[0], X2[0]*X1[1], X2[0]*X1[2],
            X2[1]*X1[0], X2[1]*X1[1], X2[1]*X1[2],
            X2[2]*X1[0], X2[2]*X1[1], X2[2]*X1[2]
        ])
    A = np.array(A)

    # Résolution par SVD (On résoud AF = 0 où A permet de décrire l'équation X1^T * F * X2)
    U, S, Vt = np.linalg.svd(A)
    F = Vt[-1].reshape(3, 3)

    # Contraindre F à être de rang 2 (en annulant la plus petite valeur singulière)
    Uf, Sf, Vtf = np.linalg.svd(F)
    Sf[-1] = 0
    F = Uf @ np.diag(Sf) @ Vtf

    # Normalisation de F (facultatif, mais souvent utile)
    F = F / F[-1, -1]

    return F


def get_intrinsic_matrix_with_specs(image_shape):
    h, w = image_shape
    fx = 0.9 * w #(focal_mm / sensor_width_mm)=0.9 (approximation a voir en detail dans le doc)
    fy = fx  # en général même focale
    cx = w / 2
    cy = h / 2
    return np.array([[fx, 0, cx],[0, fy, cy],[0,  0,  1]])

def estimate_essential_matrix(F, K):
    """
    Estime la matrice essentielle E à partir de la matrice fondamentale F et de la matrice intrinsèque K.
    E = K^T * F * K n'oublie pas que ici on prend le mm K
    """
    K_T = np.transpose(K)
    E = K_T @ F @ K
    return E


def decompose_essential_matrix(E):
    """
    Décompose la matrice essentielle E pour obtenir la rotation R et la translation t.
    """
    U, _, Vt = np.linalg.svd(E)
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    
    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt
    t1 = U[:, 2]
    t2 = -U[:, 2]
    
    if np.linalg.det(R1) > 0:
        R = R1
        t = t1
    else:
        R = R2
        t = t2
    
    return R, t


def compute_projection_matrices(K, R, t):
    """
    Calcule les matrices de projection P et P' à partir de K, R, t.
    """
    P = K @ np.hstack((np.eye(3), np.zeros((3, 1))))  # Matrice de projection de la première caméra
    P_prime = K @ np.hstack((R, t.reshape(-1, 1)))  # Matrice de projection de la deuxième caméra
    
    return P, P_prime


def reprojection_error(X, P, x):
    """
    Calcul de l'erreur de reprojection entre un point 3D X et un point 2D projeté
    à l'aide de la matrice de projection P.
    """
    X_hom = np.append(X, 1)  # Conversion en coordonnées homogènes
    x_proj = np.dot(P, X_hom)  # Projection du point 3D
    x_proj = x_proj / x_proj[2]  # Normalisation pour obtenir les coordonnées 2D
    return x - x_proj[:2]  # Erreur de reprojection (différence entre x et la projection)

def jacobian(X, P):
    """
    Calcul du Jacobien de l'erreur de reprojection par rapport au point 3D X.
    """
    X_hom = np.append(X, 1)
    J = np.zeros((2, 3))  # Jacobien de la reprojection 2D
    # Dérivées par rapport aux coordonnées X
    J[0, 0] = P[0, 0] / X_hom[2] - P[0, 2] * X_hom[0] / (X_hom[2] ** 2)
    J[0, 1] = P[0, 1] / X_hom[2] - P[0, 2] * X_hom[1] / (X_hom[2] ** 2)
    J[1, 0] = P[1, 0] / X_hom[2] - P[1, 2] * X_hom[0] / (X_hom[2] ** 2)
    J[1, 1] = P[1, 1] / X_hom[2] - P[1, 2] * X_hom[1] / (X_hom[2] ** 2)
    return J

def triangulate_dlt(x, x_prime, P, P_prime):
    """
    Estimation initiale du point 3D par méthode DLT (linéaire).
    """
    A = np.zeros((4, 4))
    A[0] = x[0] * P[2] - P[0]
    A[1] = x[1] * P[2] - P[1]
    A[2] = x_prime[0] * P_prime[2] - P_prime[0]
    A[3] = x_prime[1] * P_prime[2] - P_prime[1]
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]  # Le vecteur propre associé à la plus petite valeur propre
    return X[:3] / X[3]  # Normalisation des coordonnées 3D

def triangulate_non_linear(x, x_prime, P, P_prime, X_init, epsilon=1e-5, max_iter=100):
    """
    Triangulation non-linéaire par optimisation (méthode de Gauss-Newton).
    """
    X_k = X_init
    for _ in range(max_iter):
        # Calcul du résidu
        r0 = reprojection_error(X_k, P, x)
        r_prime = reprojection_error(X_k, P_prime, x_prime)
        r0 = np.concatenate([r0, r_prime])
        
        # Calcul du Jacobien
        J = jacobian(X_k, P)
        J_prime = jacobian(X_k, P_prime)
        J0 = np.vstack([J, J_prime])
        
        # Vérification de la condition de la matrice J^T J
        JtJ = np.dot(J0.T, J0)
        det_JtJ = np.linalg.det(JtJ)
        
        if np.abs(det_JtJ) < 1e-6:  # Matrice presque singulière
            print("Avertissement : La matrice J^T J est presque singulière, régularisation appliquée.")
            # Ajouter une petite régularisation pour stabiliser la solution
            JtJ += np.eye(JtJ.shape[0]) * 1e-6
        
        # Calcul de la mise à jour de X
        Jtr = np.dot(J0.T, r0)
        try:
            delta_X = np.linalg.solve(JtJ, Jtr)
        except np.linalg.LinAlgError:
            print("Erreur : La matrice est singulière, impossible de résoudre le système.")
            return X_k  # Retourner la dernière estimation stable
        
        X_k1 = X_k - delta_X
        
        # Condition d'arrêt
        if np.linalg.norm(X_k1 - X_k) < epsilon:
            break
        
        X_k = X_k1

    return X_k


# Exemple d'utilisation avec des points fictifs (points correspondants dans les deux images)
l_x1 = np.array([[10, 20], [30, 40], [50, 60], [70, 80], [15, 25], [35, 45], [55, 65], [75, 85]])
l_x2 = np.array([[12, 22], [32, 42], [52, 62], [72, 82], [17, 27], [37, 47], [57, 67], [77, 87]])

# Dimensions des images (hauteur, largeur)
image_shape = (1000, 1500)

# Calcul de la matrice intrinsèque K
K = get_intrinsic_matrix_with_specs(image_shape)

# Calcul de la matrice fondamentale
F = compute_fundamental_matrix(l_x1, l_x2)
print("Matrice fondamentale F :")
print(F)

# Estimation de la matrice essentielle
E = estimate_essential_matrix(F, K)
print("Matrice essentielle E :")
print(E)

# Décomposition de la matrice essentielle
R, t = decompose_essential_matrix(E)
print("Rotation R :")
print(R)
print("Translation t :")
print(t)

# Calcul de la matrice de projection P pour la première caméra
P, P_prime = compute_projection_matrices(K, R, t)
print("Matrice de projection P_prime :")
print(P_prime)
print("Matrice de projection P :")
print(P)

# Initialisation des tableaux pour les points 3D triangulés
points_3D_dlt = []
points_3D_non_linear = []

for i in range(len(l_x1)):
    # Triangulation du point i par méthode DLT pour un couple de points
    x1 = l_x1[i]
    x2 = l_x2[i]
    X_init = triangulate_dlt(x1, x2, P, P_prime)
    points_3D_dlt.append(X_init)

    # Triangulation non-linéaire
    '''X_optimized = triangulate_non_linear(x1, x2, P, P_prime, X_init)
    points_3D_non_linear.append(X_optimized)'''

# Conversion des listes en tableaux NumPy pour un affichage plus pratique
points_3D_dlt = np.array(points_3D_dlt)
points_3D_non_linear = np.array(points_3D_non_linear)

# Affichage des résultats
print("\nPoints 3D triangulés (méthode DLT) :")
print(points_3D_dlt)

'''print("\nPoints 3D triangulés (méthode non-linéaire) :")
print(points_3D_non_linear)'''