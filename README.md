# AI-Powered Crowd Behavior and Stampede Detection System

An AI-assisted video analysis application designed to monitor crowd
activity, track people, estimate movement and pose patterns, classify
crowd behavior, and present alerts and response recommendations through
a Streamlit dashboard.

> **Project status:** Academic/prototype project. Predictions and alerts
> are decision-support outputs and must not be treated as a substitute
> for trained security personnel, emergency services, or validated
> safety systems.

## Overview

Crowd incidents can develop when movement becomes irregular, crowd
density increases, or aggressive behavior emerges. This project combines
computer vision and machine-learning components to analyze uploaded
crowd videos and provide visual and text-based monitoring support.

The main application is `analyze_crowd.py`. It uses YOLO models for
person/object detection and pose estimation, ByteTrack for tracking, an
LSTM model for temporal behavior classification, and a trained Random
Forest model for aggression scoring when the model is available. The
dashboard also includes response strategies, a crowd-management chatbot,
data logging, and model/data export functionality.

## Features

-   **Video-based analysis:** Upload supported video files (`.mp4`,
    `.avi`, `.mov`) through the Streamlit interface.
-   **Object/person detection:** Uses `yolov8n.pt` through Ultralytics
    YOLO.
-   **Pose estimation:** Uses `yolov8n-pose.pt` to obtain pose-related
    information.
-   **Multi-object tracking:** Uses ByteTrack via the `supervision`
    package to associate detections across frames.
-   **Crowd behavior prediction:** Loads `lstm_crowd_behavior.h5` to
    classify temporal behavior patterns.
-   **Aggression scoring:** Loads `aggression_model.pkl`, a Random
    Forest classifier trained by `02_train_aggression_model.py`.
-   **Monitoring and alerts:** Displays crowd-analysis information and
    maps detected behavior to response guidance.
-   **Decision support:** Includes a decision agent, a Q-learning
    component, and predefined response strategies.
-   **Chatbot assistant:** Provides crowd-status information and general
    response recommendations.
-   **Data logging:** Stores crowd, detection, tracking, and Q-learning
    information in CSV/JSON files.
-   **Text-to-speech support:** Uses `pyttsx3` for spoken notifications
    where supported by the local system.

## Technology Stack

  Area                      Technology
  ------------------------- ----------------------------
  Programming language      Python
  User interface            Streamlit
  Object detection          Ultralytics YOLOv8
  Pose estimation           YOLOv8 Pose
  Object tracking           ByteTrack (`supervision`)
  Video processing          OpenCV
  Temporal behavior model   TensorFlow / Keras LSTM
  Aggression classifier     Scikit-learn Random Forest
  Data processing           NumPy, Pandas
  Model persistence         Joblib, Keras H5
  Visualization             Matplotlib
  Text-to-speech            pyttsx3

## Repository Structure

``` text
crowd-behavior-stampede-detection/
├── analyze_crowd.py                  # Main Streamlit application
├── sample_analyze.py                 # Sample/alternate analysis script
├── 01_generate_aggression_dataset.py # Builds aggression training data
├── 02_train_aggression_model.py      # Trains and saves Random Forest model
├── prepare_lstm_data.py              # Prepares data for LSTM training
├── train_lstm.py                     # Trains and saves the LSTM model
├── requirements.txt                  # Python dependencies
├── yolov8n.pt                        # YOLO object detection weights
├── yolov8n-pose.pt                   # YOLO pose estimation weights
├── lstm_crowd_behavior.h5            # Trained LSTM model
├── aggression_model.pkl              # Trained aggression model
├── label_encoder_classes.npy         # LSTM class labels
├── X_train.npy / X_test.npy          # Prepared LSTM features
├── y_train.npy / y_test.npy          # Prepared LSTM labels
├── aggression_dataset.csv            # Aggression training dataset
├── crowd_data.csv                    # Crowd-level records
├── detection_data.csv                # Detection records
├── tracking_data.csv                 # Tracking records
├── q_learning_data.json              # Q-learning data
└── alert.mp3                         # Alert audio asset
```

The repository also contains additional generated data and model
artifacts. File availability may change as the project is updated.

## Setup and Installation

### 1. Clone the repository

``` bash
git clone https://github.com/Sravya-123-45/crowd-behavior-stampede-detection.git
cd crowd-behavior-stampede-detection
```

### 2. Create and activate a virtual environment

**Windows PowerShell:**

``` powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**

``` bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

``` bash
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install supervision
```

`supervision` is imported by the main application and should be
installed even though it is not currently listed in the repository's
`requirements.txt`.

If installation fails because of a package-version conflict, use a
compatible Python environment and install the dependencies individually
as needed.

## Run the Application

From the repository root, run:

``` bash
streamlit run analyze_crowd.py
```

Streamlit will print a local URL (commonly `http://localhost:8501`).
Open that URL in a web browser.

Upload a supported video file in the application to begin analysis. Keep
the model files and supporting label file in the repository root,
because the application loads them using relative file paths.

## Model and Data Workflow

### Aggression model

The aggression model training workflow is:

1.  Run `01_generate_aggression_dataset.py` to combine available crowd,
    detection, and tracking records and generate
    `aggression_dataset.csv`.
2.  Run `02_train_aggression_model.py` to train a Random Forest
    classifier and save it as `aggression_model.pkl`.

``` bash
python 01_generate_aggression_dataset.py
python 02_train_aggression_model.py
```

**Important:** The dataset-generation script can create synthetic values
and heuristic labels for missing fields. Therefore, the resulting model
should be regarded as experimental and should not be described as
validated on independently collected, ground-truth incident data unless
such validation has been performed.

### LSTM behavior model

The LSTM workflow uses prepared NumPy arrays:

-   `X_train.npy`
-   `X_test.npy`
-   `y_train.npy`
-   `y_test.npy`

The training script converts labels to categorical form, trains an LSTM
network, and saves `lstm_crowd_behavior.h5`.

``` bash
python train_lstm.py
```

Run `prepare_lstm_data.py` first if the prepared arrays need to be
regenerated. The expected array dimensions and label encoding must match
the model architecture and the labels used by the application.

## Data and Model Files

The `.pt`, `.h5`, `.pkl`, `.npy`, `.csv`, and `.json` files in the
repository are supporting model weights, trained models, prepared
arrays, and sample/generated data used by the application and training
scripts.

If you replace or retrain a model, keep its filename and expected
input/output format consistent with the code, or update the loading and
preprocessing code accordingly.

## Limitations and Responsible Use

-   This is a prototype and has not been established as a certified or
    production-ready public-safety system.
-   Model predictions depend on video quality, camera angle, lighting,
    occlusion, crowd conditions, and the representativeness of training
    data.
-   Aggression labels generated using heuristic or synthetic data may
    not represent real-world behavior reliably.
-   The system may produce false positives or false negatives. Human
    review is required before taking action.
-   Do not use the output as the sole basis for identifying individuals,
    making accusations, or initiating emergency interventions.
-   Follow applicable privacy, consent, data-protection, and
    surveillance laws when collecting or processing video.
-   Emergency response recommendations shown by the prototype are
    illustrative guidance, not a replacement for local emergency
    procedures.

## Future Enhancements

Potential improvements include:

-   Evaluate the models on larger, diverse, independently labelled
    datasets.
-   Report class-wise precision, recall, F1-score, confusion matrices,
    and latency.
-   Improve temporal and spatial crowd-density estimation.
-   Add configurable alert thresholds and more robust alert delivery.
-   Add tests, reproducible environment versions, and structured
    configuration.
-   Document the data sources, annotation process, and model-training
    methodology.

## Author

**Sravya Annamneedu**

B.Tech, Computer Science and Engineering\
SRM University--AP

## License

A `LICENSE` file is included in this repository. Refer to that file for
the applicable license terms.
