import numpy as np
import os
import csv
import math
import random

# =========================
# CONFIG
# =========================
NUM_SAMPLES = 500
POINTS_PER_SEGMENT = 50
NOISE_STD_RANGE = (0.0, 0.01)
ANGLE_RANGE_DEG = (30, 150)

OUTPUT_DIR = "dataset"
STROKE_DIR = os.path.join(OUTPUT_DIR, "strokes")
LABEL_DIR = os.path.join(OUTPUT_DIR, "labels")

os.makedirs(STROKE_DIR, exist_ok=True)
os.makedirs(LABEL_DIR, exist_ok=True)

# =========================
# HELPERS
# =========================
def generate_line(start, direction, length, num_points):
    t = np.linspace(0, length, num_points)
    return start + np.outer(t, direction)

def unit_vector(angle_rad):
    return np.array([math.cos(angle_rad), math.sin(angle_rad)])

# =========================
# DATA GENERATION
# =========================
metadata = []

for i in range(NUM_SAMPLES):
    # Random angle
    angle_deg = random.uniform(*ANGLE_RANGE_DEG)
    angle_rad = math.radians(angle_deg)

    # Random noise
    noise_std = random.uniform(*NOISE_STD_RANGE)

    # First segment
    p0 = np.array([0.0, 0.0])
    d1 = unit_vector(0)
    seg1 = generate_line(p0, d1, 1.0, POINTS_PER_SEGMENT)

    # Second segment (corner)
    d2 = unit_vector(angle_rad)
    seg2 = generate_line(seg1[-1], d2, 1.0, POINTS_PER_SEGMENT)

    # Combine
    stroke = np.vstack([seg1, seg2])

    # Add noise
    stroke += np.random.normal(0, noise_std, stroke.shape)

    # Labels
    labels = np.zeros(len(stroke), dtype=np.int64)
    corner_idx = POINTS_PER_SEGMENT - 1
    labels[corner_idx] = 1

    # Save
    stroke_name = f"stroke_{i:04d}.npy"
    label_name = f"stroke_{i:04d}.npy"

    np.save(os.path.join(STROKE_DIR, stroke_name), stroke)
    np.save(os.path.join(LABEL_DIR, label_name), labels)

    metadata.append([stroke_name, angle_deg, noise_std, corner_idx])

# =========================
# METADATA
# =========================
with open(os.path.join(OUTPUT_DIR, "metadata.csv"), "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["stroke_file", "corner_angle_deg", "noise_std", "corner_index"])
    writer.writerows(metadata)

print("Dataset generation complete.")
