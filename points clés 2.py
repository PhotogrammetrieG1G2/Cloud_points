import cv2
import numpy as np
import matplotlib.pyplot as plt

# === 1. Charger les images et réduire leur taille si nécessaire ===
def load_and_resize(path, max_size=800):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Erreur : image {path} introuvable.")

    height, width = img.shape
    if max(height, width) > max_size:
        scaling = max_size / max(height, width)
        img = cv2.resize(img, (int(width * scaling), int(height * scaling)))
    return img

img1 = load_and_resize(r"C:\Users\Pierre\Desktop\photos converties\image5.jpg")
img2 = load_and_resize(r"C:\Users\Pierre\Desktop\photos converties\image2.jpg")

# === 2. Initialiser SIFT avec seuil abaissé ===
sift = cv2.SIFT_create(contrastThreshold=0.001) #valeur à modifier : plus il est bas plus on percoit de détails

# === 3. Détection normale (coins forts) ===
kp1_normal, des1_normal = sift.detectAndCompute(img1, None)
kp2_normal, des2_normal = sift.detectAndCompute(img2, None)

# === 4. Détection dense (grille régulière) ===
def dense_keypoints(img, step=20):
    keypoints = [cv2.KeyPoint(x, y, step)
                 for y in range(0, img.shape[0], step)
                 for x in range(0, img.shape[1], step)]
    return keypoints

kp1_dense = dense_keypoints(img1, step=20) #modifier le step pour avoir plus ou moins de points
kp2_dense = dense_keypoints(img2, step=20)

kp1_dense, des1_dense = sift.compute(img1, kp1_dense)
kp2_dense, des2_dense = sift.compute(img2, kp2_dense)

# === 5. Fusionner keypoints et descripteurs ===
kp1 = kp1_normal + kp1_dense
kp2 = kp2_normal + kp2_dense
des1 = np.vstack((des1_normal, des1_dense))
des2 = np.vstack((des2_normal, des2_dense))

print(f"Nombre total de points image 1 : {len(kp1)}")
print(f"Nombre total de points image 2 : {len(kp2)}")

# === 6. Matcher avec un Brute-Force Matcher ===
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches = bf.match(des1, des2)

# Trier par distance (meilleurs premiers)
matches = sorted(matches, key=lambda x: x.distance)

# === 7. Appliquer RANSAC pour filtrer les bons matchs ===
# Extraire les points correspondants
pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

# Calculer une matrice fondamentale avec RANSAC
F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)

# Garder uniquement les bons matchs selon le mask
matchesMask = mask.ravel().tolist()

good_matches = [m for i, m in enumerate(matches) if matchesMask[i]]

print(f"Nombre de bons matches après RANSAC : {len(good_matches)}")

# === 8. Afficher les correspondances filtrées ===
img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

plt.figure(figsize=(20,10))
plt.imshow(img_matches)
plt.title("Correspondances filtrées avec SIFT normal + dense + RANSAC")
plt.axis('off')
plt.show()



#mettre sous la forme de fonction qui prend en argument 2 photos


