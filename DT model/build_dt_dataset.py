import os
import numpy as np
from dt_features import extract_angle_features

DATASET_DIR = "./dataset_split"

def load_split(split):
    stroke_dir = os.path.join(DATASET_DIR, split, "strokes")
    label_dir = os.path.join(DATASET_DIR, split, "labels")

    X_all, y_all = [], []

    for name in os.listdir(stroke_dir):
        stroke = np.load(os.path.join(stroke_dir, name))
        labels = np.load(os.path.join(label_dir, name))

        X, y = extract_angle_features(stroke, labels)
        X_all.append(X)
        y_all.append(y)

    return np.vstack(X_all), np.hstack(y_all)

X_train, y_train = load_split("train")
X_val, y_val     = load_split("val")
X_test, y_test   = load_split("test")

np.save("X_train.npy", X_train)
np.save("y_train.npy", y_train)
np.save("X_val.npy", X_val)
np.save("y_val.npy", y_val)
np.save("X_test.npy", X_test)
np.save("y_test.npy", y_test)

print("Decision Tree dataset built.")
