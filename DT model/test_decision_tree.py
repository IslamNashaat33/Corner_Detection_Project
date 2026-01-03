import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier

X_train = np.load("DT model\\X_train.npy")
y_train = np.load("DT model\\y_train.npy")
X_test  = np.load("DT model\\X_test.npy")
y_test  = np.load("DT model\\y_test.npy")

model = DecisionTreeClassifier(
    max_depth=3,
    min_samples_leaf=50,
    random_state=42
)

model.fit(X_train, y_train)
test_preds = model.predict(X_test)

print("===== TEST RESULTS =====")
print("Precision:", precision_score(y_test, test_preds, zero_division=0))
print("Recall:   ", recall_score(y_test, test_preds, zero_division=0))
print("F1-score: ", f1_score(y_test, test_preds, zero_division=0))
