import numpy as np
import os
import csv
import math
import random

# =========================
# CONFIG
# =========================
NUM_SAMPLES = 300
POINTS_PER_SEGMENT = 50
ANGLE_RANGE_DEG = (30, 150)

OUTPUT_DIR = "dataset_clean"
STROKE_DIR = os.path.join(OUTPUT_DIR, "strokes")
LABEL_DIR = os.path.join(OUTPUT_DIR, "labels")

os.makedirs(STROKE_DIR, exist_ok=True)
os.makedirs(LABEL_DIR, exist_ok=True)

# =========================
# HELPERS
# =========================
def unit_vector(angle_rad):
    return np.array([math.cos(angle_rad), math.sin(angle_rad)])

def generate_line(start, direction, length, num_points):
    t = np.linspace(0, length, num_points)
    return start + np.outer(t, direction)

# =========================
# DATA GENERATION
# =========================
metadata = []

for i in range(NUM_SAMPLES):
    # Random corner angle
    angle_deg = random.uniform(*ANGLE_RANGE_DEG)
    angle_rad = math.radians(angle_deg)

    # First line (horizontal)
    p0 = np.array([0.0, 0.0])
    d1 = unit_vector(0.0)
    seg1 = generate_line(p0, d1, 1.0, POINTS_PER_SEGMENT)

    # Second line (corner)
    d2 = unit_vector(angle_rad)
    seg2 = generate_line(seg1[-1], d2, 1.0, POINTS_PER_SEGMENT)

    # Combine stroke
    stroke = np.vstack([seg1, seg2])

    # Labels (single corner)
    labels = np.zeros(len(stroke), dtype=np.int64)
    corner_index = POINTS_PER_SEGMENT - 1
    labels[corner_index] = 1

    # Save
    filename = f"stroke_{i:04d}.npy"
    np.save(os.path.join(STROKE_DIR, filename), stroke)
    np.save(os.path.join(LABEL_DIR, filename), labels)

    metadata.append([filename, angle_deg, corner_index])

# =========================
# SAVE METADATA
# =========================
with open(os.path.join(OUTPUT_DIR, "metadata.csv"), "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["stroke_file", "corner_angle_deg", "corner_index"])
    writer.writerows(metadata)

print("Clean (noise-free) dataset generated successfully.")
