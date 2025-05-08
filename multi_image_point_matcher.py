import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# Paramètres
INTENSITY_THRESHOLD = 60
STEP = 2  # Écart en pixels pour les points d'échantillonnage
PATCH_SIZE = 5  # Taille du patch pour le calcul du SSD

def load_and_resize(path, max_size=800):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Erreur : image {path} introuvable.")
    height, width = img.shape
    if max(height, width) > max_size:
        scaling = max_size / max(height, width)
        img = cv2.resize(img, (int(width * scaling), int(height * scaling)))
    return img

def mask_dark_pixels(img):
    return img < INTENSITY_THRESHOLD

def dense_keypoints(img):
    mask = mask_dark_pixels(img)
    keypoints = [
        cv2.KeyPoint(x, y, STEP)
        for y in range(0, img.shape[0], STEP)
        for x in range(0, img.shape[1], STEP)
        if mask[y, x]
    ]
    return keypoints

def filter_keypoints_by_intensity(img, keypoints):
    return [kp for kp in keypoints if img[int(kp.pt[1]), int(kp.pt[0])] < INTENSITY_THRESHOLD]

def obtain_correspondences(image_folder):
    image_paths = [os.path.join(image_folder, f) for f in sorted(os.listdir(image_folder)) if f.lower().endswith(".jpeg")]
    print("Images détectées :", image_paths)

    images = [load_and_resize(path) for path in image_paths]
    sift = cv2.SIFT_create(contrastThreshold=0.001)

    keypoints = []
    descriptors = []

    for img in images:
        kp_normal, des_normal = sift.detectAndCompute(img, None)
        kp_normal = filter_keypoints_by_intensity(img, kp_normal)
        if kp_normal:
            kp_normal, des_normal = sift.compute(img, kp_normal)
        else:
            des_normal = np.empty((0, 128), dtype=np.float32)

        kp_dense = dense_keypoints(img)
        if kp_dense:
            kp_dense, des_dense = sift.compute(img, kp_dense)
        else:
            des_dense = np.empty((0, 128), dtype=np.float32)

        kp = kp_normal + kp_dense
        des = np.vstack((des_normal, des_dense)) if des_normal.size and des_dense.size else (
            des_normal if des_dense.size == 0 else des_dense
        )

        keypoints.append(kp)
        descriptors.append(des)

    l1, l2 = [], []

    for i in range(len(images) - 1):
        if descriptors[i].size == 0 or descriptors[i + 1].size == 0:
            continue

        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = bf.match(descriptors[i], descriptors[i + 1])
        matches = sorted(matches, key=lambda x: x.distance)

        pts1 = np.float32([keypoints[i][m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        pts2 = np.float32([keypoints[i + 1][m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        if len(matches) >= 8:
            F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)
            if mask is not None:
                good_matches = [m for idx, m in enumerate(matches) if mask[idx]]
                l1.extend([keypoints[i][m.queryIdx].pt for m in good_matches])
                l2.extend([keypoints[i + 1][m.trainIdx].pt for m in good_matches])
            else:
                continue

        # Visualisation optionnelle
        img_matches = cv2.drawMatches(images[i], keypoints[i], images[i + 1], keypoints[i + 1], good_matches, None, flags=2)
        plt.figure(figsize=(15, 8))
        plt.title(f"Correspondances entre image {i} et {i + 1}")
        plt.imshow(img_matches, cmap='gray')
        plt.axis('off')
        plt.show()

    return l1, l2
