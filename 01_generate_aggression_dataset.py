import pandas as pd
import numpy as np
import os

# ---------------- CONFIG ---------------- #
CROWD_CSV = "crowd_data.csv"
DETECTION_CSV = "detection_data.csv"
TRACKING_CSV = "tracking_data.csv"
OUTPUT_CSV = "aggression_dataset.csv"

np.random.seed(42)

def normalize_columns(df):
    """Lowercase and standardize column names."""
    df.columns = [c.strip().lower() for c in df.columns]
    return df

frames = pd.DataFrame()

# ========== LOAD CROWD_DATA.CSV (OPTIONAL) ==========
if os.path.exists(CROWD_CSV):
    try:
        crowd = pd.read_csv(CROWD_CSV)
        crowd = normalize_columns(crowd)

        # crowd_data.csv columns you have: ['Frame', 'Density', 'Speed', 'Behavior']
        # After normalization → ['frame','density','speed','behavior']

        # Rename speed → mean_speed
        if "speed" in crowd.columns:
            crowd = crowd.rename(columns={"speed": "mean_speed"})

        # frame MUST exist
        if "frame" not in crowd.columns:
            # attempt fix for 'Frame'
            if "frame" not in crowd.columns:
                raise KeyError("crowd_data.csv has no 'frame' column")

        frames = crowd.copy()
        print("Crowd data loaded successfully:", frames.columns.tolist())
    except Exception as e:
        print("Failed to load crowd_data.csv:", e)

# ========== LOAD DETECTION_DATA.CSV ==========
if os.path.exists(DETECTION_CSV):
    try:
        det = pd.read_csv(DETECTION_CSV)
        det = normalize_columns(det)

        print("Detection data:", det.columns.tolist())

        if "frame" not in det.columns:
            raise KeyError("detection_data.csv has no 'frame' column")

        # Compute per-frame aggregates
        det_frame = det.groupby("frame").agg(
            num_people=("track_id", "nunique") if "track_id" in det.columns else ("frame", "count"),
            mean_speed=("speed", "mean") if "speed" in det.columns else ("frame", "mean"),
            speed_var=("speed", "var") if "speed" in det.columns else ("frame", "var")
        ).reset_index()

        # merge OR adopt if frames empty
        frames = det_frame if frames.empty else pd.merge(
            frames, det_frame, on="frame", how="outer"
        )

    except Exception as e:
        print("Failed processing detection_data.csv:", e)

# ========== LOAD TRACKING_DATA.CSV ==========
if os.path.exists(TRACKING_CSV):
    try:
        trk = pd.read_csv(TRACKING_CSV)
        trk = normalize_columns(trk)

        print("Tracking data:", trk.columns.tolist())

        if "frame" not in trk.columns:
            raise KeyError("tracking_data.csv has no 'frame' column")

        trk_frame = trk.groupby("frame").agg(
            trk_mean_speed=("speed", "mean"),
            trk_speed_var=("speed", "var")
        ).reset_index()

        frames = trk_frame if frames.empty else pd.merge(
            frames, trk_frame, on="frame", how="outer"
        )

    except Exception as e:
        print("Failed processing tracking_data.csv:", e)

# ========== IF ANY REQUIRED COLUMNS MISSING, SYNTHESIZE THEM ==========
# Normalize again after merges
frames = normalize_columns(frames)

if "frame" not in frames.columns:
    raise KeyError("Final merged dataset has no 'frame' column. Check CSV integrity.")

# Fill missing columns with synthetic values
if "num_people" not in frames.columns:
    frames["num_people"] = np.random.randint(0, 80, size=len(frames))

if "mean_speed" not in frames.columns:
    frames["mean_speed"] = np.random.uniform(0, 25, size=len(frames))

if "speed_var" not in frames.columns:
    frames["speed_var"] = np.random.uniform(0, 200, size=len(frames))

# Density
if "density" not in frames.columns:
    frames["density"] = frames["num_people"] / 10000.0 + np.random.normal(0, 0.0005, len(frames))

# Pose Variance
frames["pose_variance"] = np.clip(
    np.random.beta(a=2, b=5, size=len(frames)) * 2.0,
    0.0, 2.0
)

# Movement uniformity (0 = chaotic, 1 = same direction)
frames["movement_uniformity"] = np.clip(
    np.random.beta(a=2, b=2, size=len(frames)),
    0.0, 1.0
)

# ========== SYNTHETIC LABELING FUNCTION ==========
def label_row(row):
    s_var = row["speed_var"]
    dens = row["density"]
    mu = row["movement_uniformity"]
    pv = row["pose_variance"]

    score = (
        0.35 * np.tanh((s_var - 60) / 40) +
        0.25 * np.tanh((dens - 0.01) / 0.008) +
        0.25 * np.tanh((pv - 0.4) / 0.3) +
        0.15 * (1 - mu)
    )

    score += np.random.normal(0, 0.1)  # noise
    prob = 1 / (1 + np.exp(-score))
    return "Aggressive" if prob > 0.6 else "Calm"

frames["aggression_label"] = frames.apply(label_row, axis=1)

# ========== SAVE FINAL CSV ==========
final_cols = [
    "frame",
    "num_people",
    "density",
    "mean_speed",
    "speed_var",
    "pose_variance",
    "movement_uniformity",
    "aggression_label"
]

frames = frames.fillna(method="ffill").fillna(method="bfill").fillna(0.0)
frames[final_cols].to_csv(OUTPUT_CSV, index=False)

print("\nSaved:", OUTPUT_CSV)
print("Rows:", len(frames))
print(frames[final_cols].head())
