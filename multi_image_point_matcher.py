import cv2
import numpy as np
import itertools
import os
import matplotlib.pyplot as plt


INTENSITY_THRESHOLD = 180
IMAGE_FOLDER = "Images_test/Croco"



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
    """Retourne un masque des pixels sombres (< INTENSITY_THRESHOLD)."""
    return img < INTENSITY_THRESHOLD

def dense_keypoints(img, step=5):
    mask = mask_dark_pixels(img)
    keypoints = [
        cv2.KeyPoint(x, y, step)
        for y in range(0, img.shape[0], step)
        for x in range(0, img.shape[1], step)
        if mask[y, x]
    ]
    return keypoints

def filter_keypoints_by_intensity(img, keypoints):
    return [kp for kp in keypoints if img[int(kp.pt[1]), int(kp.pt[0])] < INTENSITY_THRESHOLD]

def obtain_correspondances():
    image_paths = [os.path.join(IMAGE_FOLDER, f) for f in sorted(os.listdir(IMAGE_FOLDER)) if f.lower().endswith(".jpeg")]
    print("Images détectées :", image_paths)

    images = [load_and_resize(path) for path in image_paths]
    sift = cv2.SIFT_create(contrastThreshold=0.001)

    keypoints = []
    descriptors = []

    # Détection des points clés (SIFT normal + dense filtrés sur le noir)
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

    # Initialisation des pistes
    tracks = []
    point_to_track = {}

    def has_conflict(track, new_point):
        return any(p[0] == new_point[0] for p in track)

    def add_match(img1_idx, kp1_idx, img2_idx, kp2_idx):
        p1 = (img1_idx, kp1_idx)
        p2 = (img2_idx, kp2_idx)
        t1 = point_to_track.get(p1)
        t2 = point_to_track.get(p2)

        if t1 is not None and t2 is not None:
            if t1 != t2:
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
            new_track_index = len(tracks)
            tracks.append([p1, p2])
            point_to_track[p1] = new_track_index
            point_to_track[p2] = new_track_index

    # Comparaison entre toutes les paires d’images
    for i, j in itertools.combinations(range(len(images)), 2):
        if descriptors[i].size == 0 or descriptors[j].size == 0:
            continue

        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = bf.match(descriptors[i], descriptors[j])
        matches = sorted(matches, key=lambda x: x.distance)

        pts1 = np.float32([keypoints[i][m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        pts2 = np.float32([keypoints[j][m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        if len(matches) >= 8:
            F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)
            if mask is not None:
                good_matches = [m for idx, m in enumerate(matches) if mask[idx]]
            else:
                good_matches = []
        else:
            good_matches = []

        for m in good_matches:
            add_match(i, m.queryIdx, j, m.trainIdx)

    # Nettoyage : on garde uniquement les pistes avec au moins 2 points
    tracks = [t for t in tracks if len(t) >= 2]

    l1 = []
    l2 = []
    for track in tracks:
        for i in range(len(track) - 1):
            img1_idx, kp1_idx = track[i]
            img2_idx, kp2_idx = track[i + 1]
            pt1 = keypoints[img1_idx][kp1_idx].pt
            pt2 = keypoints[img2_idx][kp2_idx].pt
            l1.append(pt1)
            l2.append(pt2)

    print(f"Nombre total de pistes multi-vues : {len(tracks)}")

    # Visualisation des correspondances pour les paires d'images successives
    for i in range(len(images) - 1):
        img1 = images[i]
        img2 = images[i + 1]
        kp1 = keypoints[i]
        kp2 = keypoints[i + 1]

        if descriptors[i].size == 0 or descriptors[i + 1].size == 0:
            continue

        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = bf.match(descriptors[i], descriptors[i + 1])
        matches = sorted(matches, key=lambda x: x.distance)

        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        if len(matches) >= 8:
            F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)
            matchesMask = mask.ravel().tolist() if mask is not None else [0] * len(matches)
        else:
            matchesMask = [0] * len(matches)

        draw_params = dict(matchColor=(0, 255, 0),
                           singlePointColor=None,
                           matchesMask=matchesMask,
                           flags=2)

        img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, **draw_params)
        plt.figure(figsize=(15, 8))
        plt.title(f"Correspondances entre image {i} et {i + 1}")
        plt.imshow(img_matches, cmap='gray')
        plt.axis('off')
        plt.show()

    return l1, l2

# === Lancement ===
# l1, l2 = obtain_correspondances()
# print(len(l1), "correspondances trouvées.")
# print(l1)
# print(l2)