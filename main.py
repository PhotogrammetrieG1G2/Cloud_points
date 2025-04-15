import numpy as np

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

l_x1 = [[10, 20], [30, 40], [50, 60], [70, 80], [15, 25], [35, 45], [55, 65], [75, 85]]
l_x2 = [[12, 22], [32, 42], [52, 62], [72, 82], [17, 27], [37, 47], [57, 67], [77, 87]]

F = fondamental_matrix(l_x1, l_x2)
print(F)
