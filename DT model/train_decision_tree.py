import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, recall_score, f1_score

X_train = np.load("DT model\\X_train.npy")
y_train = np.load("DT model\\y_train.npy")
X_val   = np.load("DT model\\X_val.npy")
y_val   = np.load("DT model\\y_val.npy")

model = DecisionTreeClassifier(
    max_depth=3,          # very important
    min_samples_leaf=50,  # avoid overfitting
    random_state=42
)

model.fit(X_train, y_train)

val_preds = model.predict(X_val)

print("Validation results:")
print("Precision:", precision_score(y_val, val_preds, zero_division=0))
print("Recall:   ", recall_score(y_val, val_preds, zero_division=0))
print("F1-score: ", f1_score(y_val, val_preds, zero_division=0))
