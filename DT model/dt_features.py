import numpy as np

def compute_angle(p_prev, p, p_next):
    v1 = p - p_prev
    v2 = p_next - p

    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)

    if n1 == 0 or n2 == 0:
        return 0.0

    cos_theta = np.dot(v1, v2) / (n1 * n2)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.arccos(cos_theta)

def extract_angle_features(stroke, labels):
    X = []
    y = []

    for i in range(1, len(stroke) - 1):
        angle = compute_angle(stroke[i-1], stroke[i], stroke[i+1])
        X.append([angle])
        y.append(labels[i])

    return np.array(X), np.array(y)
