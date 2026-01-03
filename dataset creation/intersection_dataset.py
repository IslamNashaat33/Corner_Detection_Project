import numpy as np
import os
import csv
import math
import random

# =========================
# CONFIG
# =========================
NUM_SINGLE_CORNER = 200
NUM_INTERSECTIONS = 100

POINTS_PER_SEGMENT = 100
ANGLE_RANGE_DEG = (30, 150)

OUTPUT_DIR = "dataset_clean_intersections"
STROKE_DIR = os.path.join(OUTPUT_DIR, "strokes")
LABEL_DIR = os.path.join(OUTPUT_DIR, "labels")

os.makedirs(STROKE_DIR, exist_ok=True)
os.makedirs(LABEL_DIR, exist_ok=True)

# =========================
# HELPERS
# =========================
def unit_vector(angle):
    return np.array([math.cos(angle), math.sin(angle)])

def generate_line(center, direction, half_length, num_points):
    t = np.linspace(-half_length, half_length, num_points)
    return center + np.outer(t, direction)

# =========================
# DATA GENERATION
# =========================
metadata = []
stroke_id = 0

# ---------- 1) Single-stroke CORNER samples ----------
for _ in range(NUM_SINGLE_CORNER):
    angle_deg = random.uniform(*ANGLE_RANGE_DEG)
    angle_rad = math.radians(angle_deg)

    p0 = np.array([0.0, 0.0])
    d1 = unit_vector(0.0)
    d2 = unit_vector(angle_rad)

    seg1 = generate_line(p0, d1, 1.0, POINTS_PER_SEGMENT // 2)
    seg2 = generate_line(seg1[-1], d2, 1.0, POINTS_PER_SEGMENT // 2)

    stroke = np.vstack([seg1, seg2])

    labels = np.zeros(len(stroke), dtype=np.int64)
    labels[len(seg1) - 1] = 1  # true corner

    filename = f"stroke_{stroke_id:04d}.npy"
    np.save(os.path.join(STROKE_DIR, filename), stroke)
    np.save(os.path.join(LABEL_DIR, filename), labels)

    metadata.append([filename, "single_corner", angle_deg])
    stroke_id += 1

# ---------- 2) Two-stroke INTERSECTION samples (NO corners) ----------
for _ in range(NUM_INTERSECTIONS):
    center = np.array([0.0, 0.0])

    angle1 = random.uniform(0, math.pi)
    angle2 = angle1 + math.pi / 2  # perpendicular intersection

    d1 = unit_vector(angle1)
    d2 = unit_vector(angle2)

    # Stroke A
    stroke_a = generate_line(center, d1, 1.2, POINTS_PER_SEGMENT)
    labels_a = np.zeros(len(stroke_a), dtype=np.int64)

    filename_a = f"stroke_{stroke_id:04d}.npy"
    np.save(os.path.join(STROKE_DIR, filename_a), stroke_a)
    np.save(os.path.join(LABEL_DIR, filename_a), labels_a)
    metadata.append([filename_a, "intersection", "A"])
    stroke_id += 1

    # Stroke B
    stroke_b = generate_line(center, d2, 1.2, POINTS_PER_SEGMENT)
    labels_b = np.zeros(len(stroke_b), dtype=np.int64)

    filename_b = f"stroke_{stroke_id:04d}.npy"
    np.save(os.path.join(STROKE_DIR, filename_b), stroke_b)
    np.save(os.path.join(LABEL_DIR, filename_b), labels_b)
    metadata.append([filename_b, "intersection", "B"])
    stroke_id += 1

# =========================
# SAVE METADATA
# =========================
with open(os.path.join(OUTPUT_DIR, "metadata.csv"), "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["stroke_file", "type", "info"])
    writer.writerows(metadata)

print("Dataset with true intersections (no corners) generated successfully.")
