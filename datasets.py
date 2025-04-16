import cv2
import numpy as np



def random_points_on_cube(n):
    points = []
    face_normals = [
        (1, 0, 0), (-1, 0, 0),
        (0, 1, 0), (0, -1, 0),
        (0, 0, 1), (0, 0, -1)
    ]

    for _ in range(n):
        normal = face_normals[np.random.choice(len(face_normals))]
        u, v = np.random.uniform(-1, 1, size=2)
        z = np.random.uniform(-20, 1, size=1)
        axis = np.argmax(np.abs(normal))
        point = np.array([u, v, z[0]])
        point[axis] = normal[axis]  # fixer un axe à -1 ou 1 selon la face
        #point[2] += 2  # translation en profondeur
        points.append(-point)

    return np.array(points, dtype=np.float32)

def random_points_on_circle(n, radius=4.0):
    theta = np.random.uniform(0, 2 * np.pi, n)
    phi = np.random.uniform(0, np.pi, n)
    
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)
    points = np.stack((x, y, z), axis=1)

    return points.astype(np.float32)

def random_points_on_pyramid(n, height=15.0):
    points = []

    # Sommets de la base carrée
    base = np.array([
        [-1, -1, -1],
        [ 1, -1, -1],
        [ 1,  1, -1],
        [-1,  1, -1],
    ])
    
    # Sommet de la pyramide
    apex = np.array([0, 0, -height])

    # Les 4 faces triangulaires (apex, p1, p2) et celle de la base (qui est carré, donc on la découpe en 2 triangles)
    faces = [
        (apex, base[0], base[1]),
        (apex, base[1], base[2]),
        (apex, base[2], base[3]),
        (apex, base[3], base[0])
    ]

    for _ in range(n):
        # Choisir une face au hasard
        v0, v1, v2 = faces[np.random.choice(len(faces))]

        if not any(np.array_equal(apex, v) for v in [v0, v1, v2]):
            u, v = np.random.uniform(-2, 2, 2)
            point = np.array([u, v, 0])
            point = apex
        else:
            # Tirage barycentrique uniforme
            u, v = np.random.uniform(0, 1, 2)
            if u + v > 1:
                u, v = 1 - u, 1 - v
            point = (1 - u - v) * v0 + u * v1 + v * v2

        points.append(point)

    return np.array(points, dtype=np.float32)


def get_intrinsic_matrix_with_specs(image_shape):
    h, w = image_shape
    fx = 0.9 * w #(focal_mm / sensor_width_mm)=0.9 (approximation a voir en detail dans le doc)
    fy = fx  # en général même focale
    cx = w / 2
    cy = h / 2
    return np.array([[fx, 0, cx],[0, fy, cy],[0,  0,  1]])

def generate_dataset(num_points):
    # On génère num_points points à la surface d’un cube (6 faces)
    
    points_3D_pyramid = random_points_on_pyramid(num_points)
    points_3D_cube = random_points_on_cube(3*num_points)
    points_3D = np.vstack((points_3D_pyramid, points_3D_cube))

    num_points_final = 3*num_points + 1*num_points

    # Matrice intrinsèque
    K = get_intrinsic_matrix_with_specs(image_shape = (1000, 1500))

    # Caméra 1 (identité)
    R1 = np.eye(3)
    t1 = np.zeros((3, 1))
    P1 = K @ np.hstack((R1, t1))

    # Caméra 2 (translation + légère rotation)
    R2, _ = cv2.Rodrigues(np.array([0.05, -0.1, 0.05]))
    t2 = np.array([[-0.5], [0.0], [0.0]])
    P2 = K @ np.hstack((R2, t2))

    # Projection
    points_3D_hom = np.hstack((points_3D, np.ones((num_points_final, 1))))
    pts1_hom = (P1 @ points_3D_hom.T).T
    pts2_hom = (P2 @ points_3D_hom.T).T

    pts1 = pts1_hom[:, :2] / pts1_hom[:, 2, np.newaxis]
    pts2 = pts2_hom[:, :2] / pts2_hom[:, 2, np.newaxis]

    return pts1, pts2