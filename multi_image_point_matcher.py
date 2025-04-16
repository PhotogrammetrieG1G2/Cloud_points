import cv2                                                                    #traitement d'images et de vidéos
import numpy as np
import itertools

# --- Étape 1 : Chargement des images ---
image_paths = ['img1.jpg', 'img2.jpg', 'img3.jpg']                            #liste des chemins des images à traiter
images = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in image_paths]     #charge chaque image en niveau de gris

# --- Étape 2 : Détection des keypoints et descripteurs ---
sift = cv2.SIFT_create()                                                      #détecteur de points d'intérêt SIFT (scale-invariant feature transform)
keypoints = []                                                                #points d'intérêt
descriptors = []                                                              #matrice numpy contenant des vecteurs décrivant la texture autour du point

for img in images:
    kp, des = sift.detectAndCompute(img, None)
    keypoints.append(kp)
    descriptors.append(des)

# --- Étape 3 : Structures de suivi ---
tracks = []                                                                   #liste des pistes multi-vues ie des mêmes points entre toutes les images
point_to_track = {}                                                           #dictionnaire permettant de retrouver rapidement dans quelle piste se trouve un point donné

def add_match(img1_idx, kp1_idx, img2_idx, kp2_idx):                          #fonction qui prend une correspondance entre deux points de deux images
    p1 = (img1_idx, kp1_idx)
    p2 = (img2_idx, kp2_idx)
    
    t1 = point_to_track.get(p1)
    t2 = point_to_track.get(p2)
    
    if t1 is not None and t2 is not None:                                     #cas les deux points sont déjà dans deux pistes => on les fusionne
        if t1 != t2:
            # Fusionner deux pistes
            tracks[t1].extend(tracks[t2])
            for pt in tracks[t2]:
                point_to_track[pt] = t1
            tracks[t2] = []
    elif t1 is not None:                                                      #cas un seul point est déjà dans une piste => on ajoute l'autre
        tracks[t1].append(p2)
        point_to_track[p2] = t1
    elif t2 is not None:                                                      
        tracks[t2].append(p1)
        point_to_track[p1] = t2
    else:                                                                     #cas aucun des deux points n'est encore suivi => on crée une nouvelle piste
        new_track_index = len(tracks)
        tracks.append([p1, p2])
        point_to_track[p1] = new_track_index
        point_to_track[p2] = new_track_index

# --- Étape 4 : Appel à pt_corres (fonction qui trouve les points de correspondances entre deux images) pour toutes les paires d'images ---
def pt_corres(img1_idx, img2_idx): ####### à modifier #########
    raise NotImplementedError("")

for i, j in itertools.combinations(range(len(images)), 2):
    matches = pt_corres(i, j)  
    for idx1, idx2 in matches:
        add_match(i, idx1, j, idx2)

# --- Étape 5 : Nettoyage des pistes vides ---
tracks = [t for t in tracks if len(t) >= 2]

# --- Étape 6 : Résumé ---
print(f"Nombre total de pistes multi-vues : {len(tracks)}")
for i, track in enumerate(tracks[:5]):  # Affiche les 5 premières pistes
    print(f"Piste {i+1} : {track}")
