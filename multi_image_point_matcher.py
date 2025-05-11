import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# ====================== PARAMÈTRES ======================

# Seuil d’intensité pour les pixels utilisables
INTENSITY_THRESHOLD = 100

# Échantillonnage dense
STEP = 2
PATCH_SIZE = 2

# Paramètres globaux du SIFT (détection très dense)
SIFT_PARAMS = {
    "nfeatures": 0,              # Illimité
    "nOctaveLayers": 5,          # Plus de niveaux d’échelle
    "contrastThreshold": 0.001, # Très sensible
    "edgeThreshold": 2,          # Accepte bords fins
    "sigma": 1.2                 # Flou initial léger
}

# ====================== FONCTIONS ======================

def load_and_resize(path, max_size=800):
    img_color = cv2.imread(path, cv2.IMREAD_COLOR)
    if img_color is None:
        raise ValueError(f"Erreur : image {path} introuvable.")
    height, width = img_color.shape[:2]
    if max(height, width) > max_size:
        scale = max_size / max(height, width)
        new_size = (int(width * scale), int(height * scale))
        img_color = cv2.resize(img_color, new_size)
    return img_color

def mask_dark_pixels(img_gray):
    return img_gray < INTENSITY_THRESHOLD

def dense_keypoints(img_gray):
    mask = mask_dark_pixels(img_gray)
    return [
        cv2.KeyPoint(x, y, STEP)
        for y in range(0, img_gray.shape[0], STEP)
        for x in range(0, img_gray.shape[1], STEP)
        if mask[y, x]
    ]

def filter_keypoints_by_intensity(img_gray, keypoints):
    return [kp for kp in keypoints if img_gray[int(kp.pt[1]), int(kp.pt[0])] < INTENSITY_THRESHOLD]

def get_color_at(img, pt):
    x, y = int(round(pt[0])), int(round(pt[1]))
    if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
        bgr = img[y, x]
        return bgr[::-1] / 255.0  # RGB normalisé
    return [0.5, 0.5, 0.5]

# ====================== PIPELINE ======================

def obtain_correspondences(image_paths):
    print("Images détectées :", image_paths)

    # Initialisation
    sift = cv2.SIFT_create(**SIFT_PARAMS)
    images_color = [load_and_resize(path) for path in image_paths]
    keypoints, descriptors = [], []

    # Détection des keypoints & descripteurs
    for img_color in images_color:
        img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

        kp_normal, des_normal = sift.detectAndCompute(img_gray, None)
        kp_normal = filter_keypoints_by_intensity(img_gray, kp_normal)
        des_normal = sift.compute(img_gray, kp_normal)[1] if kp_normal else np.empty((0, 128), dtype=np.float32)

        kp_dense = dense_keypoints(img_gray)
        des_dense = sift.compute(img_gray, kp_dense)[1] if kp_dense else np.empty((0, 128), dtype=np.float32)

        kp_combined = kp_normal + kp_dense
        des_combined = np.vstack([d for d in [des_normal, des_dense] if d.size > 0])

        keypoints.append(kp_combined)
        descriptors.append(des_combined)

    # Initialisation des résultats
    l1, l2, colors_pixels, F_matrices = [], [], [], []

    # Appariement entre paires successives
    i = 0
    for i in range(len(images_color) - 1):
        if descriptors[i].size == 0 or descriptors[i + 1].size == 0:
            continue

        bf = cv2.BFMatcher()
        raw_matches = bf.knnMatch(descriptors[i], descriptors[i + 1], k=2)
        matches = [m for m, n in raw_matches if m.distance < 0.75 * n.distance]

        if len(matches) < 8:
            continue

        pts1 = np.float32([keypoints[i][m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        pts2 = np.float32([keypoints[i + 1][m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)
        if F is None or mask is None:
            continue

        F_matrices.append(F)
        mask = mask.ravel().astype(bool)
        good_matches = [m for m, valid in zip(matches, mask) if valid]

        for m in good_matches:
            pt1 = keypoints[i][m.queryIdx].pt
            pt2 = keypoints[i + 1][m.trainIdx].pt
            l1.append(pt1)
            l2.append(pt2)
            colors_pixels.append(get_color_at(images_color[i], pt1))

        # Points voisins (densification autour des bons matches)
        for m in tqdm(good_matches, desc=f"Image {i} voisins"):
            pt1 = np.array(keypoints[i][m.queryIdx].pt)
            pt2 = np.array(keypoints[i + 1][m.trainIdx].pt)
            delta = pt2 - pt1

            for dx in range(-PATCH_SIZE, PATCH_SIZE + 1):
                for dy in range(-PATCH_SIZE, PATCH_SIZE + 1):
                    if dx == 0 and dy == 0:
                        continue
                    pt1_neigh = pt1 + [dx, dy]
                    pt2_neigh = pt1_neigh + delta

                    l1.append(tuple(pt1_neigh))
                    l2.append(tuple(pt2_neigh))
                    colors_pixels.append(get_color_at(images_color[i], pt1_neigh))

        # Affichage des correspondances
        img1_gray = cv2.cvtColor(images_color[i], cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(images_color[i + 1], cv2.COLOR_BGR2GRAY)
        img_matches = cv2.drawMatches(
            img1_gray, keypoints[i], img2_gray, keypoints[i + 1],
            good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        plt.figure(figsize=(15, 8))
        plt.title(f"Correspondances entre image {i} et {i + 1}")
        plt.imshow(img_matches, cmap='gray')
        plt.axis('off')
        plt.show()

    return l1, l2, F_matrices, colors_pixels
