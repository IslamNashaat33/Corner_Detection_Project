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

def compute_curvature(p_prev, p, p_next):
    a = np.linalg.norm(p - p_prev)
    b = np.linalg.norm(p_next - p)
    c = np.linalg.norm(p_next - p_prev)

    if a * b * c == 0:
        return 0.0

    area = abs(
        (p_prev[0]*(p[1]-p_next[1]) +
         p[0]*(p_next[1]-p_prev[1]) +
         p_next[0]*(p_prev[1]-p[1])) / 2.0
    )

    return (4 * area) / (a * b * c)

def extract_features(stroke):
    N = len(stroke)
    features = []

    for i in range(N):
        if i == 0 or i == N - 1:
            features.append([0.0, 0.0])
            continue

        angle = compute_angle(stroke[i-1], stroke[i], stroke[i+1])
        curvature = compute_curvature(stroke[i-1], stroke[i], stroke[i+1])
        features.append([angle, curvature])

    return np.array(features, dtype=np.float32)
