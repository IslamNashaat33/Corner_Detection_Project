import os
import shutil
import random

# =========================
# CONFIG
# =========================
SOURCE_DIR = "dataset_intersections_v2"

SPLIT_DIR = "dataset_split_v2"

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

assert TRAIN_RATIO + VAL_RATIO + TEST_RATIO == 1.0

random.seed(42)

# =========================
# PATHS
# =========================
stroke_src = os.path.join(SOURCE_DIR, "strokes")
label_src = os.path.join(SOURCE_DIR, "labels")

# Create output folders
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(SPLIT_DIR, split, "strokes"), exist_ok=True)
    os.makedirs(os.path.join(SPLIT_DIR, split, "labels"), exist_ok=True)

# =========================
# SPLIT FILES
# =========================
files = sorted(os.listdir(stroke_src))
random.shuffle(files)

n_total = len(files)
n_train = int(n_total * TRAIN_RATIO)
n_val = int(n_total * VAL_RATIO)

train_files = files[:n_train]
val_files = files[n_train:n_train + n_val]
test_files = files[n_train + n_val:]

def copy_files(file_list, split):
    for name in file_list:
        shutil.copy(
            os.path.join(stroke_src, name),
            os.path.join(SPLIT_DIR, split, "strokes", name)
        )
        shutil.copy(
            os.path.join(label_src, name),
            os.path.join(SPLIT_DIR, split, "labels", name)
        )

copy_files(train_files, "train")
copy_files(val_files, "val")
copy_files(test_files, "test")

# =========================
# SUMMARY
# =========================
print("Dataset split complete:")
print(f"  Train: {len(train_files)}")
print(f"  Val:   {len(val_files)}")
print(f"  Test:  {len(test_files)}")
