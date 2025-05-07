import cv2
import numpy as np
import matplotlib as plt
import itertools
import sys
import os

import points_cles

# Vérifie qu’un dossier est passé en argument
if len(sys.argv) < 2:
    print("Usage: python script.py <dossier_images>")
    sys.exit(1)

image_folder = sys.argv[1]

# Liste les fichiers JPG dans le dossier
image_paths = [os.path.join(image_folder, f) for f in sorted(os.listdir(image_folder)) if f.lower().endswith(".jpg")]

print("Images détectées :", image_paths)

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
def has_conflict(track, new_point):
    return any(p[0] == new_point[0] for p in track)

def add_match(img1_idx, kp1_idx, img2_idx, kp2_idx):
    p1 = (img1_idx, kp1_idx)
    p2 = (img2_idx, kp2_idx)
    
    t1 = point_to_track.get(p1)
    t2 = point_to_track.get(p2)

    if t1 is not None and t2 is not None:
        if t1 != t2:
            # Vérifie qu'on ne fusionne pas des points dupliqués dans une même image
            track1 = tracks[t1]
            track2 = tracks[t2]
            if not any(p[0] in [q[0] for q in track1] for p in track2):
                tracks[t1].extend(track2)
                for pt in track2:
                    point_to_track[pt] = t1
                tracks[t2] = []
    elif t1 is not None:
        if not has_conflict(tracks[t1], p2):
            tracks[t1].append(p2)
            point_to_track[p2] = t1
    elif t2 is not None:
        if not has_conflict(tracks[t2], p1):
            tracks[t2].append(p1)
            point_to_track[p1] = t2
    else:
        # Nouvelle piste
        new_track_index = len(tracks)
        tracks.append([p1, p2])
        point_to_track[p1] = new_track_index
        point_to_track[p2] = new_track_index


# === Étape 4 : Trouve les correspondances entre deux images (indices des keypoints) ===
def pt_corres(img1_idx, img2_idx):
    matches, kp1, kp2 = points_cles.points_cles(image_paths[img1_idx], image_paths[img2_idx])
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