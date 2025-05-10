import cv2
import numpy as np
import matplotlib.pyplot as plt
import tqdm

# Paramètres
INTENSITY_THRESHOLD = 100
STEP = 2
PATCH_SIZE = 15

'''
def load_and_resize(path, max_size=800):
    img_gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img_color = cv2.imread(path)
    if img_gray is None or img_color is None:
        raise ValueError(f"Erreur : image {path} introuvable.")
    height, width = img_gray.shape
    if max(height, width) > max_size:
        scaling = max_size / max(height, width)
        new_size = (int(width * scaling), int(height * scaling))
        img_gray = cv2.resize(img_gray, new_size)
        img_color = cv2.resize(img_color, new_size)
    return img_gray, img_color
'''

def load_and_resize(path, max_size=800):
    img_color = cv2.imread(path, cv2.IMREAD_COLOR)
    if img_color is None:
        raise ValueError(f"Erreur : image {path} introuvable.")
    height, width = img_color.shape[:2]
    if max(height, width) > max_size:
        scaling = max_size / max(height, width)
        new_size = (int(width * scaling), int(height * scaling))
        img_color = cv2.resize(img_color, new_size)
    return img_color


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

def obtain_correspondences(image_paths):
    print("Images détectées :", image_paths)

    # Chargement des images couleur
    images_color = [load_and_resize(path) for path in image_paths]

    # Initialisation de SIFT
    sift = cv2.SIFT_create(contrastThreshold=0.001)

    keypoints, descriptors = [], []

    # Détection de points clés et descripteurs pour chaque image
    for img_color in images_color:
        img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

        # Points SIFT normaux
        kp_normal, des_normal = sift.detectAndCompute(img_gray, None)
        kp_normal = filter_keypoints_by_intensity(img_gray, kp_normal)
        if kp_normal:
            kp_normal, des_normal = sift.compute(img_gray, kp_normal)
        else:
            des_normal = np.empty((0, 128), dtype=np.float32)

        # Points denses
        kp_dense = dense_keypoints(img_gray)
        if kp_dense:
            kp_dense, des_dense = sift.compute(img_gray, kp_dense)
        else:
            des_dense = np.empty((0, 128), dtype=np.float32)

        kp = kp_normal + kp_dense
        des = np.vstack((des_normal, des_dense)) if des_normal.size and des_dense.size else (
            des_normal if des_dense.size == 0 else des_dense)

        keypoints.append(kp)
        descriptors.append(des)

    l1, l2, colors_pixels, F_matrixs = [], [], [], []

    for i in range(len(images_color) - 1):
        if descriptors[i].size == 0 or descriptors[i + 1].size == 0:
            continue

        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = sorted(bf.match(descriptors[i], descriptors[i + 1]), key=lambda x: x.distance)

        pts1 = np.float32([keypoints[i][m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        pts2 = np.float32([keypoints[i + 1][m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        if len(matches) < 8:
            continue

        F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)
        if F is None or mask is None:
            continue

        F_matrixs.append(F)
        good_matches = [m for idx, m in enumerate(matches) if mask[idx]]

        # Ajout des correspondances principales
        for m in good_matches:
            pt1 = keypoints[i][m.queryIdx].pt
            pt2 = keypoints[i + 1][m.trainIdx].pt
            l1.append(pt1)
            l2.append(pt2)

            x, y = int(round(pt1[0])), int(round(pt1[1]))
            if 0 <= x < images_color[i].shape[1] and 0 <= y < images_color[i].shape[0]:
                bgr = images_color[i][y, x]
                rgb = bgr[::-1] / 255.0
            else:
                rgb = [0.5, 0.5, 0.5]
            colors_pixels.append(rgb)

        # Ajout des voisins autour de pt1
        for m in tqdm.tqdm(good_matches):
            pt1 = np.array(keypoints[i][m.queryIdx].pt)
            pt2 = np.array(keypoints[i + 1][m.trainIdx].pt)
            delta = pt2 - pt1  # vecteur de déplacement

            x, y = pt1
            neighbors = [
                (nx, ny)
                for nx in range(int(x - PATCH_SIZE), int(x + PATCH_SIZE) + 1)
                for ny in range(int(y - PATCH_SIZE), int(y + PATCH_SIZE) + 1)
                if (nx, ny) != (x, y)
            ]

            for pt1_neigh in neighbors:
                pt1_neigh = np.array(pt1_neigh)
                pt2_neigh = pt1_neigh + delta  # on applique le déplacement estimé

                l1.append(tuple(pt1_neigh))
                l2.append(tuple(pt2_neigh))

                nx, ny = int(round(pt1_neigh[0])), int(round(pt1_neigh[1]))
                if 0 <= nx < images_color[i].shape[1] and 0 <= ny < images_color[i].shape[0]:
                    bgr = images_color[i][ny, nx]
                    rgb = bgr[::-1] / 255.0
                else:
                    rgb = [0.5, 0.5, 0.5]
                colors_pixels.append(rgb)


        # Affichage des correspondances
        img_gray1 = cv2.cvtColor(images_color[i], cv2.COLOR_BGR2GRAY)
        img_gray2 = cv2.cvtColor(images_color[i + 1], cv2.COLOR_BGR2GRAY)
        img_matches = cv2.drawMatches(img_gray1, keypoints[i], img_gray2, keypoints[i + 1], good_matches, None, flags=2)

        plt.figure(figsize=(15, 8))
        plt.title(f"Correspondances entre image {i} et {i + 1}")
        plt.imshow(img_matches, cmap='gray')
        plt.axis('off')
        plt.show()

    return l1, l2, F_matrixs, colors_pixels
