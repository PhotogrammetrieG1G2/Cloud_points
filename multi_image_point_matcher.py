import cv2
import numpy as np
import matplotlib as plt
import itertools
import sys
import os

# Vérifie qu’un dossier est passé en argument
if len(sys.argv) < 2:
    print("Usage: python script.py <dossier_images>")
    sys.exit(1)

image_folder = sys.argv[1]

# Liste les fichiers JPG dans le dossier
image_paths = [os.path.join(image_folder, f) for f in sorted(os.listdir(image_folder)) if f.lower().endswith(".jpg")]

print("Images détectées :", image_paths)


# === Fonction utilitaire : charger et redimensionner une image ===
def load_and_resize(path, max_size=800):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # Charge l'image en niveaux de gris
    if img is None:
        raise ValueError(f"Erreur : image {path} introuvable.")
    height, width = img.shape
    if max(height, width) > max_size:  # Réduit l'image si elle dépasse la taille maximale
        scaling = max_size / max(height, width)
        img = cv2.resize(img, (int(width * scaling), int(height * scaling)))
    return img

# === Générer une grille régulière de points-clés (dense) ===
def dense_keypoints(img, step=20):
    keypoints = [cv2.KeyPoint(x, y, step)
                 for y in range(0, img.shape[0], step)
                 for x in range(0, img.shape[1], step)]
    return keypoints

# === Fonction principale pour détecter et matcher les points-clés entre deux images ===
def points_clés(chemin1, chemin2):
    # Chargement et redimensionnement des deux images
    img1 = load_and_resize(chemin1)
    img2 = load_and_resize(chemin2)
    
    # Initialisation du détecteur SIFT avec seuil bas pour plus de détection
    sift = cv2.SIFT_create(contrastThreshold=0.001)
    
    # Détection SIFT classique (keypoints + descripteurs)
    kp1_normal, des1_normal = sift.detectAndCompute(img1, None)
    kp2_normal, des2_normal = sift.detectAndCompute(img2, None)
    
    # Détection dense (grille régulière)
    kp1_dense = dense_keypoints(img1, step=20)
    kp2_dense = dense_keypoints(img2, step=20)
    kp1_dense, des1_dense = sift.compute(img1, kp1_dense)
    kp2_dense, des2_dense = sift.compute(img2, kp2_dense)
    
    # Fusion des points normaux et denses
    kp1 = kp1_normal + kp1_dense
    kp2 = kp2_normal + kp2_dense
    des1 = np.vstack((des1_normal, des1_dense))
    des2 = np.vstack((des2_normal, des2_dense))
    
    # Appariement avec Brute Force Matcher
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    
    # Estimation de la matrice fondamentale pour filtrer les bons matchs
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)
    
    # On garde uniquement les correspondances valides (inliers)
    matchesMask = mask.ravel().tolist()
    good_matches = [m for i, m in enumerate(matches) if matchesMask[i]]
    return good_matches, kp1, kp2

# === Étape 1 : Chargement des images à traiter ===
#image_paths = ['img1.jpg', 'img2.jpg', 'img3.jpg']
images = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in image_paths]  # Chargement en niveaux de gris

# === Étape 2 : Détection simple de keypoints/descripteurs pour suivi plus tard ===
sift = cv2.SIFT_create()
keypoints = []    # Liste des points-clés pour chaque image
descriptors = []  # Liste des descripteurs

for img in images:
    kp, des = sift.detectAndCompute(img, None)
    keypoints.append(kp)
    descriptors.append(des)

# === Étape 3 : Structures pour le suivi multi-vues ===
tracks = []            # Liste des pistes (chaînes de correspondance inter-images)
point_to_track = {}    # Dictionnaire pour savoir à quelle piste appartient un point (image_idx, kp_idx)

# Ajoute une correspondance à la bonne piste
def add_match(img1_idx, kp1_idx, img2_idx, kp2_idx):
    p1 = (img1_idx, kp1_idx)
    p2 = (img2_idx, kp2_idx)
    
    t1 = point_to_track.get(p1)
    t2 = point_to_track.get(p2)
    
    if t1 is not None and t2 is not None:
        if t1 != t2:  # Fusion de pistes si nécessaire
            tracks[t1].extend(tracks[t2])
            for pt in tracks[t2]:
                point_to_track[pt] = t1
            tracks[t2] = []  # On vide l'ancienne piste fusionnée
    elif t1 is not None:
        tracks[t1].append(p2)
        point_to_track[p2] = t1
    elif t2 is not None:
        tracks[t2].append(p1)
        point_to_track[p1] = t2
    else:
        # Crée une nouvelle piste
        new_track_index = len(tracks)
        tracks.append([p1, p2])
        point_to_track[p1] = new_track_index
        point_to_track[p2] = new_track_index

# === Étape 4 : Trouve les correspondances entre deux images (indices des keypoints) ===
def pt_corres(img1_idx, img2_idx):
    matches, kp1, kp2 = points_clés(image_paths[img1_idx], image_paths[img2_idx])
    match_indices = []

    for m in matches:
        pt1 = m.queryIdx
        pt2 = m.trainIdx

        # On récupère les coordonnées des points appariés
        pt1_coord = tuple(np.round(kp1[pt1].pt))
        pt2_coord = tuple(np.round(kp2[pt2].pt))

        # On cherche l'indice du point d'origine le plus proche (dans keypoints d'origine)
        idx1 = min(range(len(keypoints[img1_idx])), key=lambda i: np.linalg.norm(np.array(keypoints[img1_idx][i].pt) - pt1_coord))
        idx2 = min(range(len(keypoints[img2_idx])), key=lambda i: np.linalg.norm(np.array(keypoints[img2_idx][i].pt) - pt2_coord))
        
        match_indices.append((idx1, idx2))
    return match_indices

# === Boucle sur toutes les paires d’images pour créer les pistes ===
for i, j in itertools.combinations(range(len(images)), 2):
    matches = pt_corres(i, j)  # Liste de tuples (idx1, idx2)
    for idx1, idx2 in matches:
        add_match(i, idx1, j, idx2)

# === Étape 5 : On enlève les pistes vides ou trop courtes ===
tracks = [t for t in tracks if len(t) >= 2]

# === Étape 6 : Affichage du résumé des résultats ===
print(f"Nombre total de pistes multi-vues : {len(tracks)}")
for i, track in enumerate(tracks[:5]):  # Affiche les 5 premières pistes
    print(f"Piste {i+1} : {track}")

## Visualisation des points de correspondance

# def afficher_correspondances(img1_path, img2_path):
#     matches, kp1, kp2 = points_clés(img1_path, img2_path)
#     img1 = cv2.imread(img1_path)
#     img2 = cv2.imread(img2_path)
#     img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None, flags=2)
#     cv2.imshow("Correspondances", img_matches)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# afficher_correspondances('img1.jpg', 'img2.jpg')