import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os

DATA_CSV = "aggression_dataset.csv"
MODEL_PATH = "aggression_model.pkl"

if not os.path.exists(DATA_CSV):
    raise FileNotFoundError(f"{DATA_CSV} not found. Run 01_generate_aggression_dataset.py first.")

df = pd.read_csv(DATA_CSV)

# Features & label
feature_cols = [
    "num_people",
    "density",
    "mean_speed",
    "speed_var",
    "pose_variance",
    "movement_uniformity"
]

X = df[feature_cols].values
y_raw = df["aggression_label"].values

# Encode labels to 0/1
label_map = {"Calm": 0, "Aggressive": 1}
y = np.array([label_map.get(v, 0) for v in y_raw])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    class_weight="balanced",
    random_state=42,
)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

joblib.dump({"model": clf, "label_map": label_map, "feature_cols": feature_cols}, MODEL_PATH)
print(f"Saved aggression model to {MODEL_PATH}")
