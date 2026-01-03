import numpy as np
import matplotlib.pyplot as plt
import os
import random

# =========================
# CONFIG
# =========================
DATASET_DIR = "dataset_intersections_v2"  # path to dataset
STROKE_DIR = os.path.join(DATASET_DIR, "strokes")
LABEL_DIR = os.path.join(DATASET_DIR, "labels")

NUM_SAMPLES_TO_VIEW = 30     # how many strokes to show
GRID_COLS = 4                # grid columns
RANDOM_SAMPLE = False         # False = first N samples
OFFSET = 300               # offset to start viewing from (None = middle of dataset, or set a specific number)

# =========================
# LOAD FILE NAMES
# =========================
stroke_files = sorted(os.listdir(STROKE_DIR))
total_files = len(stroke_files)

if RANDOM_SAMPLE:
    stroke_files = random.sample(stroke_files, min(NUM_SAMPLES_TO_VIEW, total_files))
else:
    # Calculate offset (default to middle if None)
    if OFFSET is None:
        offset = max(0, (total_files - NUM_SAMPLES_TO_VIEW) // 2)
    else:
        offset = max(0, min(OFFSET, total_files - NUM_SAMPLES_TO_VIEW))
    
    stroke_files = stroke_files[offset:offset + NUM_SAMPLES_TO_VIEW]

num_samples = len(stroke_files)
rows = (num_samples + GRID_COLS - 1) // GRID_COLS

# =========================
# PLOT GRID
# =========================
plt.figure(figsize=(4 * GRID_COLS, 4 * rows))

for i, stroke_name in enumerate(stroke_files):
    stroke = np.load(os.path.join(STROKE_DIR, stroke_name))
    labels = np.load(os.path.join(LABEL_DIR, stroke_name))

    ax = plt.subplot(rows, GRID_COLS, i + 1)

    # Plot stroke
    ax.plot(stroke[:, 0], stroke[:, 1], '-o', markersize=2)

    # Plot corner points
    corner_idx = np.where(labels == 1)[0]
    ax.scatter(
        stroke[corner_idx, 0],
        stroke[corner_idx, 1],
        color='red',
        s=40,
        label='Corner'
    )

    ax.set_title(stroke_name, fontsize=9)
    ax.axis('equal')
    ax.axis('off')

plt.tight_layout()
plt.show()
