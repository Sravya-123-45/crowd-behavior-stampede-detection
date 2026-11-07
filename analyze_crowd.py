import streamlit as st
from ultralytics import YOLO
import cv2
import supervision as sv
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import tempfile
import os
import random
import math
import csv
import time
from joblib import load as joblib_load

import pyttsx3
tts_engine = pyttsx3.init()

# Optional: Adjust voice speed & volume
tts_engine.setProperty('rate', 170)      # speed
tts_engine.setProperty('volume', 1.0)    # max volume

def speak(text):
    try:
        tts_engine.say(text)
        tts_engine.runAndWait()
    except Exception as e:
        print("TTS Error:", e)


# ---------------------- PAGE CONFIG ---------------------- #
st.set_page_config(
    page_title="AI Crowd Behavior Analysis",
    layout="wide",
)


# ---------------------- SAFE DETECTIONS FALLBACK ---------------------- #
class SafeDetections:
    def __init__(self):
        self.xyxy = []
        self.tracker_id = []

    def __len__(self):
        return 0


# ---------------------- MODEL LOADING ---------------------- #
@st.cache_resource
def load_models():
    yolo_model = YOLO("yolov8n.pt")   # Object detection model
    pose_model = YOLO("yolov8n-pose.pt")  # Pose detection model

    # Disable any internal Ultralytics tracker (we use our own)
    for model in (yolo_model, pose_model):
        if hasattr(model, "predictor"):
            if hasattr(model.predictor, "args"):
                model.predictor.args.tracker = None
            if hasattr(model.predictor, "trackers"):
                model.predictor.trackers = []

    try:
        lstm_model = load_model("lstm_crowd_behavior.h5")
    except Exception as e:
        st.error(f"Error: Could not load lstm_crowd_behavior.h5: {e}. Please place the model file in this folder.")
        return None, None, None

    return yolo_model, pose_model, lstm_model


yolo_model, pose_model, lstm_model = load_models()
if yolo_model is None or pose_model is None or lstm_model is None:
    st.stop()

# Load label encoder classes for LSTM
try:
    label_encoder_classes = np.load("label_encoder_classes.npy", allow_pickle=True)
except Exception as e:
    st.error(f"Error: Could not load label_encoder_classes.npy: {e}")
    st.stop()


# Load aggression ML model (trained from aggression_dataset.csv)
try:
    aggression_model = joblib_load("aggression_model.pkl")
except Exception:
    aggression_model = None
    st.warning("aggression_model.pkl not found or failed to load. ML aggression scoring will be disabled.")


tracker = sv.ByteTrack()
sequence_length = 10


# ---------------------- RESPONSE STRATEGIES ---------------------- #
RESPONSE_STRATEGIES = {
    "Calm": "Maintain regular monitoring. No immediate action required.",
    "Aggressive": "Deploy additional security personnel. Prepare for potential conflict de-escalation.",
    "Dispersing": "Monitor exits and ensure clear pathways. Consider crowd flow management.",
    "Stampede": "Activate emergency protocols immediately. Coordinate with local authorities and medical teams."
}


# ---------------------- Q-LEARNING AGENT ---------------------- #
class QLearningAgent:
    def __init__(self, actions, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.1):
        self.q_table = {}
        self.actions = actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate

    def get_state(self, density, speed, variance, movement_uniformity, pose_variance):
        density_bin = int(min(density * 1000, 50) // 10)
        speed_bin = int(min(speed, 50) // 10)
        variance_bin = int(min(variance, 100) // 20)
        movement_bin = int(min(movement_uniformity * 100, 100) // 20)
        pose_bin = int(min(pose_variance * 100, 100) // 20)
        return (density_bin, speed_bin, variance_bin, movement_bin, pose_bin)

    def get_q_value(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        q_values = [self.get_q_value(state, a) for a in self.actions]
        return self.actions[int(np.argmax(q_values))]

    def update_q_table(self, state, action, reward, next_state):
        old_q = self.get_q_value(state, action)
        next_max_q = max([self.get_q_value(next_state, a) for a in self.actions])
        new_q = old_q + self.lr * (reward + self.gamma * next_max_q - old_q)
        self.q_table[(state, action)] = new_q


# ---------------------- DECISION AGENT ---------------------- #
class DecisionAgent:
    def __init__(self):
        self.alert_level = "Normal"
        self.last_escalation_frame = 0
        self.escalation_cooldown = 300  # frames

    def decide_action(self, rule_behavior, lstm_behavior, frame_count):
        current_behavior = lstm_behavior if lstm_behavior != "Unknown" else rule_behavior
        if frame_count - self.last_escalation_frame < self.escalation_cooldown:
            return None

        if current_behavior == "Stampede":
            self.alert_level = "Critical"
            self.last_escalation_frame = frame_count
            return "Activate emergency protocols: Notify authorities, activate sirens, and dispatch medical teams."
        elif current_behavior == "Aggressive":
            self.alert_level = "High"
            self.last_escalation_frame = frame_count
            return "Escalate alert: Notify security personnel and prepare for de-escalation."
        elif current_behavior == "Dispersing":
            self.alert_level = "Moderate"
            self.last_escalation_frame = frame_count
            return "Monitor situation: Ensure clear pathways and adjust crowd flow."
        else:
            self.alert_level = "Normal"
            return None


# ---------------------- CHATBOT ---------------------- #
class CrowdManagementChatbot:
    def __init__(self):
        self.responses = {
            "greeting": [
                "Hello! I'm your crowd management assistant. How can I help you today?",
                "Welcome to the crowd monitoring system. What information do you need?"
            ],
            "status": [
                "Currently monitoring {num_people} people. Behavior status: {behavior}.",
                "I'm tracking {num_people} individuals with {behavior} behavior pattern."
            ],
            "help": [
                "I can provide crowd status updates, alert notifications, and recommendations for crowd management.",
                "You can ask me about current crowd density, behavior patterns, or request specific actions."
            ],
            "action_request": {
                "Calm": [
                    "No action needed at this time. Continuing normal monitoring.",
                    "Regular monitoring is sufficient. The crowd is calm."
                ],
                "Dispersing": [
                    "Monitoring exit pathways. Consider opening additional exits to facilitate flow.",
                    "Recommended action: Guide crowd toward available exits and ensure paths remain clear."
                ],
                "Aggressive": [
                    "ALERT: Deploying security personnel to sections with aggressive behavior.",
                    "ALERT: Recommend immediate conflict de-escalation procedures in affected areas."
                ],
                "Stampede": [
                    "EMERGENCY: Activating all emergency protocols. Authorities have been notified.",
                    "EMERGENCY: Coordinating with emergency services. Medical teams dispatched."
                ]
            },
            "recommendation": {
                "Calm": [
                    "Continue regular monitoring with standard personnel.",
                    "Maintain normal operations. No special measures needed."
                ],
                "Dispersing": [
                    "Ensure all exits are unobstructed and visible.",
                    "Consider increasing lighting near exits and deploying staff to guide crowd movement."
                ],
                "Aggressive": [
                    "Deploy conflict resolution specialists to affected areas.",
                    "Consider separating agitated individuals and establishing calm zones."
                ],
                "Stampede": [
                    "Immediately open all emergency exits. Activate all alarm systems.",
                    "Deploy all available personnel to guide crowd and prevent bottlenecks."
                ]
            },
            "general": [
                "I'm monitoring the crowd in real-time and will alert you to any significant changes.",
                "My analysis is based on object detection, pose estimation, and behavior prediction models."
            ]
        }

        self.intents = {
            "greeting": ["hello", "hi", "hey", "greetings", "good morning", "good afternoon", "good evening"],
            "status": ["status", "update", "what's happening", "current situation", "what is going on"],
            "help": ["help", "what can you do", "options", "capabilities", "commands", "how to use"],
            "action": ["what should", "what action", "what to do", "recommend action", "next steps", "how to respond"],
            "recommendation": ["recommend", "suggest", "advice", "what would you suggest", "best practice"]
        }

    def recognize_intent(self, message):
        message = message.lower()
        for intent, patterns in self.intents.items():
            for pattern in patterns:
                if pattern in message:
                    return intent
        return "general"

    def get_response(self, message, crowd_data):
        intent = self.recognize_intent(message)
        behavior = crowd_data.get("behavior", "Calm")
        num_people = crowd_data.get("num_people", 0)

        if intent == "greeting":
            return random.choice(self.responses["greeting"])
        elif intent == "status":
            return random.choice(self.responses["status"]).format(
                num_people=num_people, behavior=behavior
            )
        elif intent == "help":
            return random.choice(self.responses["help"])
        elif intent == "action":
            return random.choice(self.responses["action_request"][behavior])
        elif intent == "recommendation":
            return random.choice(self.responses["recommendation"][behavior])
        else:
            return random.choice(self.responses["general"])


# ---------------------- HELPERS: CSV LOGGING ---------------------- #
def append_rows_to_csv(file_path, header, rows):
    file_exists = os.path.exists(file_path)
    with open(file_path, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerows(rows)


def save_q_table_to_json(file_path, q_table):
    import json
    with open(file_path, "w") as f:
        json.dump({str(k): v for k, v in q_table.items()}, f)


# ---------------------- SESSION STATE ---------------------- #
actions = [
    "increase_density", "decrease_density",
    "increase_speed", "decrease_speed",
    "increase_variance", "decrease_variance",
    "increase_pose", "decrease_pose"
]

if "rl_agent" not in st.session_state:
    st.session_state.rl_agent = QLearningAgent(actions)

if "decision_agent" not in st.session_state:
    st.session_state.decision_agent = DecisionAgent()

if "chatbot" not in st.session_state:
    st.session_state.chatbot = CrowdManagementChatbot()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "assistant",
         "content": "Welcome to the Crowd Management Assistant. Ask me about status, actions, or recommendations."}
    ]

if "current_crowd_data" not in st.session_state:
    st.session_state.current_crowd_data = {
        "num_people": 0, "density": 0, "behavior": "Calm"
    }

if "running" not in st.session_state:
    st.session_state.running = False


# Global thresholds
thresholds = {"density": 0.015, "speed": 25, "variance": 50, "pose_variance": 0.5}


# ---------------------- UI: MAIN TITLE ---------------------- #
st.title("AI-Powered Crowd Behavior Predictor with Movement, Pose & Aggression Detection")
st.write("Analyze crowd behavior with real-time anomaly heatmaps, overlays, movement, pose detection, and aggression alerts.")


# ---------------------- TABS ---------------------- #
tab1, tab2 = st.tabs(["Analysis View", "Chatbot Assistant"])


# ---------------------- TAB 1: ANALYSIS ---------------------- #
with tab1:
    # Input selection
    input_option = st.radio("Select Input Source:", ("Real-Time Drone/CCTV Feed", "Pre-Recorded Video"))

    def generate_heatmap(frame, centroids, intensity_factor=50):
        heatmap = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)
        for x, y in centroids:
            cv2.circle(heatmap, (x, y), 30, intensity_factor, -1)
        heatmap = cv2.GaussianBlur(heatmap, (51, 51), 0)
        heatmap = np.clip(heatmap, 0, 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        return heatmap_color

    def process_frame(frame,
                      prev_positions,
                      density_history,
                      speed_history,
                      pose_history,
                      time_history,
                      frame_count,
                      rl_agent,
                      thresholds):
        global tracker, yolo_model, pose_model, lstm_model, label_encoder_classes, aggression_model

        # YOLO person detection
        results = yolo_model(frame)
        filtered_boxes = [box for box in results[0].boxes if int(box.cls) == 0]

        if len(filtered_boxes) == 0:
            detections = sv.Detections(
                xyxy=np.zeros((0, 4), dtype=np.float32),
                confidence=np.array([], dtype=np.float32),
                class_id=np.array([], dtype=np.int32),
            )
        else:
            detections = sv.Detections(
                xyxy=np.array([box.xyxy[0].cpu().numpy() for box in filtered_boxes]),
                confidence=np.array([box.conf[0].cpu().numpy() for box in filtered_boxes]),
                class_id=np.array([0] * len(filtered_boxes)),
            )

        tracked_detections = tracker.update_with_detections(detections)
        if (tracked_detections is None or
                getattr(tracked_detections, "xyxy", None) is None or
                getattr(tracked_detections, "tracker_id", None) is None):
            tracked_detections = SafeDetections()

        annotated_frame = frame.copy()

        # Pose detection
        pose_results = pose_model(frame)
        pose_keypoints = pose_results[0].keypoints.xy.cpu().numpy() if pose_results[0].keypoints is not None else []

        num_people = len(tracked_detections)
        frame_area = frame.shape[0] * frame.shape[1]
        density = num_people / frame_area * 10000 if frame_area > 0 else 0

        speeds = []
        centroids = []
        movement_vectors = []
        pose_orientations = []

        detection_rows = []
        tracking_rows = []

        # Tracking
        for box, track_id in zip(tracked_detections.xyxy, tracked_detections.tracker_id):
            x1, y1, x2, y2 = map(int, box)
            centroid = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            centroids.append(centroid)

            speed = 0.0
            if prev_positions[track_id] is not None:
                prev_x, prev_y = prev_positions[track_id]
                dx = centroid[0] - prev_x
                dy = centroid[1] - prev_y
                speed = float(np.sqrt(dx**2 + dy**2))
                speeds.append(speed)
                angle = math.atan2(dy, dx) if speed > 0 else 0
                movement_vectors.append((speed, angle))
                arrow_length = min(int(speed * 2), 50)
                end_x = int(centroid[0] + arrow_length * math.cos(angle))
                end_y = int(centroid[1] + arrow_length * math.sin(angle))
                cv2.arrowedLine(annotated_frame, centroid, (end_x, end_y), (255, 255, 0), 2, tipLength=0.3)

            prev_positions[track_id] = centroid
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"ID: {int(track_id)}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            detection_rows.append([
                frame_count, int(track_id), x1, y1, x2, y2, f"{speed:.2f}", ""  # behavior placeholder
            ])
            tracking_rows.append([
                frame_count, int(track_id), centroid[0], centroid[1], f"{speed:.2f}"
            ])

        # Pose processing + raised hands heuristic
        raised_hand_count = 0
        pose_person_count = 0

        for keypoints in pose_keypoints:
            if len(keypoints) >= 17:
                left_shoulder = keypoints[5][:2]
                right_shoulder = keypoints[6][:2]
                left_hip = keypoints[11][:2]
                right_hip = keypoints[12][:2]
                left_wrist = keypoints[9][:2]
                right_wrist = keypoints[10][:2]

                if all(np.all(kp != 0) for kp in [left_shoulder, right_shoulder, left_hip, right_hip]):
                    torso_dx = (right_shoulder[0] + right_hip[0]) / 2 - (left_shoulder[0] + left_hip[0]) / 2
                    torso_dy = (right_shoulder[1] + right_hip[1]) / 2 - (left_shoulder[1] + left_hip[1]) / 2
                    orientation = math.atan2(torso_dy, torso_dx)
                    pose_orientations.append(orientation)

                    # Draw keypoints
                    for kp in keypoints:
                        if np.all(kp != 0):
                            cv2.circle(annotated_frame, (int(kp[0]), int(kp[1])), 4, (255, 0, 255), -1)

                    # Raised hands: wrists above shoulder line (remember: y grows downward)
                    pose_person_count += 1
                    raised = False
                    if np.all(left_wrist != 0) and left_wrist[1] < left_shoulder[1] - 10:
                        raised = True
                    if np.all(right_wrist != 0) and right_wrist[1] < right_shoulder[1] - 10:
                        raised = True
                    if raised:
                        raised_hand_count += 1

        avg_speed = float(np.mean(speeds)) if speeds else 0.0
        speed_variance = float(np.var(speeds)) if speeds else 0.0
        pose_variance = float(np.var(pose_orientations)) if pose_orientations else 0.0

        if movement_vectors:
            angles = [vec[1] for vec in movement_vectors]
            angle_variance = np.var([math.cos(a) for a in angles]) + np.var([math.sin(a) for a in angles])
            movement_uniformity = max(0.0, 1.0 - float(angle_variance))
        else:
            movement_uniformity = 0.0

        if pose_person_count > 0:
            hands_raised_ratio = raised_hand_count / pose_person_count
        else:
            hands_raised_ratio = 0.0

        # RL threshold adjustment
        state = rl_agent.get_state(density, avg_speed, speed_variance, movement_uniformity, pose_variance)
        action = rl_agent.choose_action(state)

        if action == "increase_density":
            thresholds["density"] += 0.001
        elif action == "decrease_density":
            thresholds["density"] = max(0.001, thresholds["density"] - 0.001)
        elif action == "increase_speed":
            thresholds["speed"] += 1
        elif action == "decrease_speed":
            thresholds["speed"] = max(5, thresholds["speed"] - 1)
        elif action == "increase_variance":
            thresholds["variance"] += 5
        elif action == "decrease_variance":
            thresholds["variance"] = max(10, thresholds["variance"] - 5)
        elif action == "increase_pose":
            thresholds["pose_variance"] += 0.05
        elif action == "decrease_pose":
            thresholds["pose_variance"] = max(0.1, thresholds["pose_variance"] - 0.05)

        # ML aggression model (crowd-level features)
        aggression_score = None
        aggression_flag = False
        if aggression_model is not None:
            try:
                features = np.array([[
                    density,
                    avg_speed,
                    speed_variance,
                    pose_variance,
                    movement_uniformity
                ]], dtype=float)
                if hasattr(aggression_model, "predict_proba"):
                    prob = aggression_model.predict_proba(features)[0][1]
                    aggression_score = float(prob)
                    aggression_flag = prob >= 0.6
                else:
                    pred_label = aggression_model.predict(features)[0]
                    aggression_flag = bool(pred_label)
            except Exception:
                aggression_score = None
                aggression_flag = False

        # Pose-based aggression boost if many hands raised
        if hands_raised_ratio > 0.5:
            aggression_flag = True
            if aggression_score is None:
                aggression_score = 0.9
            else:
                aggression_score = max(aggression_score, 0.9)

        # Rule-based behavior
        if (density > thresholds["density"] and avg_speed > thresholds["speed"] and
                speed_variance > thresholds["variance"] and movement_uniformity > 0.8 and
                pose_variance < thresholds["pose_variance"]):
            rule_behavior = "Stampede"
        elif (density > thresholds["density"] * 0.66 and avg_speed > thresholds["speed"] * 0.8 and
              movement_uniformity < 0.4 and pose_variance > thresholds["pose_variance"]):
            rule_behavior = "Aggressive"
        elif aggression_flag:
            # If ML / pose says aggression, upgrade behavior
            rule_behavior = "Aggressive"
        elif density < thresholds["density"] * 0.33 and avg_speed > thresholds["speed"] * 0.4:
            rule_behavior = "Dispersing"
        else:
            rule_behavior = "Calm"

        # LSTM behavior prediction
        lstm_behavior = "Unknown"
        if len(density_history) >= sequence_length - 1:
            density_history.append(density)
            speed_history.append(avg_speed)
            pose_history.append(pose_variance)
            seq = np.array(list(zip(
                density_history[-sequence_length:],
                speed_history[-sequence_length:],
                pose_history[-sequence_length:]
            )))
            seq = seq.reshape(1, sequence_length, 3)
            pred = lstm_model.predict(seq, verbose=0)
            lstm_behavior = label_encoder_classes[int(np.argmax(pred))]
        else:
            density_history.append(density)
            speed_history.append(avg_speed)
            pose_history.append(pose_variance)

        frame_count += 1
        time_history.append(frame_count)

        # RL feedback
        reward = 1 if lstm_behavior == rule_behavior and lstm_behavior != "Unknown" else -1
        next_state = rl_agent.get_state(density, avg_speed, speed_variance, movement_uniformity, pose_variance)
        rl_agent.update_q_table(state, action, reward, next_state)

        # Final behavior for logs
        detected_behavior = lstm_behavior if lstm_behavior != "Unknown" else rule_behavior

        # Fill behavior column in detection rows
        for row in detection_rows:
            row[-1] = detected_behavior

        # Log to CSV continuously
        if detection_rows:
            append_rows_to_csv(
                "detection_data.csv",
                ["frame", "track_id", "x1", "y1", "x2", "y2", "speed", "behavior"],
                detection_rows
            )
        if tracking_rows:
            append_rows_to_csv(
                "tracking_data.csv",
                ["frame", "track_id", "centroid_x", "centroid_y", "speed"],
                tracking_rows
            )

        # Heatmap
        intensity = min(50 + int(density * 1000 + avg_speed), 255)
        heatmap = generate_heatmap(frame, centroids, intensity)
        annotated_frame = cv2.addWeighted(annotated_frame, 0.7, heatmap, 0.3, 0)

        # Highlight anomalies
        if rule_behavior in ["Aggressive", "Stampede"] or lstm_behavior in ["Aggressive", "Stampede"]:
            for x, y in centroids:
                cv2.circle(annotated_frame, (x, y), 40, (0, 0, 255), 3)

        # Update crowd data for chatbot
        st.session_state.current_crowd_data = {
            "num_people": num_people,
            "density": density,
            "speed": avg_speed,
            "variance": speed_variance,
            "uniformity": movement_uniformity,
            "pose_variance": pose_variance,
            "behavior": detected_behavior,
            "frame": frame_count,
            "aggression_score": aggression_score,
            "hands_raised_ratio": hands_raised_ratio
        }

        # Autonomous actions
        autonomous_action = st.session_state.decision_agent.decide_action(
            rule_behavior, lstm_behavior, frame_count
        )
        if autonomous_action:
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": f"🚨 AUTONOMOUS ACTION: {autonomous_action}",
                "alert_level": st.session_state.decision_agent.alert_level
            })

            if st.session_state.decision_agent.alert_level == "Critical":
                st.error(f"CRITICAL ALERT: {autonomous_action}")
            elif st.session_state.decision_agent.alert_level == "High":
                st.warning(f"HIGH ALERT: {autonomous_action}")
            elif st.session_state.decision_agent.alert_level == "Moderate":
                st.info(f"MODERATE ALERT: {autonomous_action}")

        # Periodic status update
        if frame_count % 150 == 0:
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": f"Status update: {detected_behavior} behavior. "
                           f"{num_people} people, density={density:.4f}, uniformity={movement_uniformity:.2f}"
            })

        # Text overlays
        cv2.putText(annotated_frame, f"Rule-Based: {rule_behavior}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.putText(annotated_frame, f"LSTM Pred: {lstm_behavior}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(annotated_frame, f"People: {num_people}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.putText(annotated_frame, f"Density: {density:.4f}", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.putText(annotated_frame, f"Speed Var: {speed_variance:.2f}", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.putText(annotated_frame, f"Move Uniformity: {movement_uniformity:.2f}", (10, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(annotated_frame, f"Pose Var: {pose_variance:.2f}", (10, 210),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(annotated_frame, f"Hands Raised Ratio: {hands_raised_ratio:.2f}", (10, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        if aggression_score is not None:
            cv2.putText(annotated_frame, f"AggScore: {aggression_score:.2f}", (10, 270),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if rule_behavior == "Aggressive" or lstm_behavior == "Aggressive":
            cv2.putText(annotated_frame, "ALERT: Aggressive Behavior", (10, 300),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        if rule_behavior == "Stampede" or lstm_behavior == "Stampede":
            cv2.putText(annotated_frame, "ALERT: STAMPEDE", (10, 330),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Plots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 9))
        fig.suptitle('Crowd Analysis Trends')
        ax1.plot(time_history, density_history, 'b-', label='Density')
        ax1.set_title('Density Over Time')
        ax1.set_xlabel('Frame')
        ax1.set_ylabel('People/10k pixels')
        ax1.legend()
        ax1.grid(True)

        ax2.plot(time_history, speed_history, 'r-', label='Speed')
        ax2.set_title('Average Speed Over Time')
        ax2.set_xlabel('Frame')
        ax2.set_ylabel('Pixels/Frame')
        ax2.legend()
        ax2.grid(True)

        ax3.plot(time_history, pose_history, 'g-', label='Pose Variance')
        ax3.set_title('Pose Variance Over Time')
        ax3.set_xlabel('Frame')
        ax3.set_ylabel('Variance')
        ax3.legend()
        ax3.grid(True)

        plt.tight_layout()

        return annotated_frame, num_people, density, avg_speed, pose_variance, rule_behavior, lstm_behavior, fig, frame_count

    # -------- REAL-TIME FEED -------- #
    if input_option == "Real-Time Drone/CCTV Feed":
        st.write("Using real-time feed from camera (default: webcam). Enter an RTSP URL for drone/CCTV if needed.")
        rtsp_url = st.text_input("RTSP URL (leave blank for webcam)", "")
        video_source = rtsp_url if rtsp_url else 0

        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            st.error("Error: Could not open real-time feed. Check your camera or RTSP URL.")
            st.stop()

        prev_positions = defaultdict(lambda: None)
        density_history = []
        speed_history = []
        pose_history = []
        time_history = []
        frame_count = 0

        video_placeholder = st.empty()
        plot_placeholder = st.empty()
        metrics_placeholder = st.empty()

        start_col, stop_col = st.columns(2)
        with start_col:
            if st.button("Start Real-Time Analysis", key="start_button"):
                st.session_state.running = True
        with stop_col:
            if st.button("Stop Real-Time Analysis", key="stop_button"):
                st.session_state.running = False

        if st.session_state.running:
            while cap.isOpened() and st.session_state.running:
                ret, frame = cap.read()
                if not ret:
                    st.warning("End of feed or error reading frame.")
                    break

                (annotated_frame, num_people, density, avg_speed,
                 pose_variance, rule_behavior, lstm_behavior,
                 fig, frame_count) = process_frame(
                    frame, prev_positions,
                    density_history, speed_history,
                    pose_history, time_history,
                    frame_count,
                    st.session_state.rl_agent,
                    thresholds
                )

                annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(annotated_frame_rgb, caption=f"Frame {frame_count}")
                plot_placeholder.pyplot(fig)
                metrics_placeholder.write(
                    f"Frame {frame_count}: People={num_people}, "
                    f"Density={density:.4f}, AvgSpeed={avg_speed:.2f}, "
                    f"PoseVar={pose_variance:.2f}, Rule={rule_behavior}, LSTM={lstm_behavior}"
                )

                if rule_behavior == "Aggressive" or lstm_behavior == "Aggressive":
                    st.warning("Aggressive behavior detected!")
                    speak("Warning. Aggression detected.")
                if rule_behavior == "Stampede" or lstm_behavior == "Stampede":
                    st.error("Stampede detected! Immediate action recommended.")
                    speak("Emergency! Stampede detected. Take action immediately.")

                plt.close(fig)

        cap.release()

    # -------- PRE-RECORDED VIDEO -------- #
    else:
        uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])
        if uploaded_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_file.read())
            video_path = tfile.name

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                st.error("Error: Could not open video.")
                st.stop()

            prev_positions = defaultdict(lambda: None)
            density_history = []
            speed_history = []
            pose_history = []
            time_history = []
            frame_count = 0

            video_placeholder = st.empty()
            plot_placeholder = st.empty()
            metrics_placeholder = st.empty()

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                (annotated_frame, num_people, density, avg_speed,
                 pose_variance, rule_behavior, lstm_behavior,
                 fig, frame_count) = process_frame(
                    frame, prev_positions,
                    density_history, speed_history,
                    pose_history, time_history,
                    frame_count,
                    st.session_state.rl_agent,
                    thresholds
                )

                annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(annotated_frame_rgb, caption=f"Frame {frame_count}")
                plot_placeholder.pyplot(fig)
                metrics_placeholder.write(
                    f"Frame {frame_count}: People={num_people}, "
                    f"Density={density:.4f}, AvgSpeed={avg_speed:.2f}, "
                    f"PoseVar={pose_variance:.2f}, Rule={rule_behavior}, LSTM={lstm_behavior}"
                )

                if rule_behavior == "Aggressive" or lstm_behavior == "Aggressive":
                    st.warning("Aggressive behavior detected!")
                if rule_behavior == "Stampede" or lstm_behavior == "Stampede":
                    st.error("Stampede detected! Immediate action recommended.")

                plt.close(fig)
                time.sleep(0.05)

            # Clean up temp file safely on Windows
            try:
                cap.release()
                cv2.destroyAllWindows()
                time.sleep(0.3)
                os.unlink(video_path)
            except PermissionError:
                try:
                    time.sleep(1)
                    os.unlink(video_path)
                except Exception:
                    pass


# ---------------------- TAB 2: CHATBOT ---------------------- #
with tab2:
    st.subheader("Crowd Management Assistant")

    st.info(
        "You can ask things like:\n"
        "- *\"What is the current status?\"*\n"
        "- *\"What should we do now?\"*\n"
        "- *\"Recommend an action\"*"
    )

    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f"**You**: {message['content']}")
            else:
                if "alert_level" in message:
                    if message["alert_level"] == "Critical":
                        st.error(f"**Assistant**: {message['content']}")
                    elif message["alert_level"] == "High":
                        st.warning(f"**Assistant**: {message['content']}")
                    elif message["alert_level"] == "Moderate":
                        st.info(f"**Assistant**: {message['content']}")
                    else:
                        st.markdown(f"**Assistant**: {message['content']}")
                else:
                    st.markdown(f"**Assistant**: {message['content']}")

    st.markdown("---")
    with st.form(key='chat_form'):
        user_message = st.text_input("Ask the assistant about crowd management:", key="chat_input")
        submit_button = st.form_submit_button(label="Send")

        if submit_button and user_message.strip():
            msg = user_message.strip()
            st.session_state.chat_history.append({"role": "user", "content": msg})
            response = st.session_state.chatbot.get_response(msg, st.session_state.current_crowd_data)
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            st.rerun()


# ---------------------- SYSTEM DASHBOARD ---------------------- #
st.header("System Dashboard")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Crowd Analysis Settings")

    with st.expander("Advanced Detection Settings"):
        detection_confidence = st.slider("Detection Confidence", 0.0, 1.0, 0.5, 0.01)
        st.info(f"Objects will be detected only if confidence is above {detection_confidence}")
        tracking_persistence = st.slider("Tracking Persistence", 1, 50, 25)
        st.info(f"Number of frames to keep track of disappeared objects: {tracking_persistence}")

    with st.expander("Behavior Detection Thresholds"):
        manual_density_threshold = st.slider("Density Threshold", 0.001, 0.05, float(thresholds["density"]), 0.001)
        manual_speed_threshold = st.slider("Speed Threshold", 5.0, 50.0, float(thresholds["speed"]), 0.5)
        manual_variance_threshold = st.slider("Variance Threshold", 10.0, 200.0, float(thresholds["variance"]), 5.0)
        manual_pose_threshold = st.slider("Pose Variance Threshold", 0.1, 2.0, float(thresholds["pose_variance"]), 0.1)

        if st.button("Update Thresholds"):
            thresholds["density"] = manual_density_threshold
            thresholds["speed"] = manual_speed_threshold
            thresholds["variance"] = manual_variance_threshold
            thresholds["pose_variance"] = manual_pose_threshold
            st.success("Thresholds updated successfully!")

with col2:
    st.subheader("Response Strategies")

    for behavior, strategy in RESPONSE_STRATEGIES.items():
        if behavior == "Stampede":
            st.error(f"**{behavior}**: {strategy}")
        elif behavior == "Aggressive":
            st.warning(f"**{behavior}**: {strategy}")
        elif behavior == "Dispersing":
            st.info(f"**{behavior}**: {strategy}")
        else:
            st.success(f"**{behavior}**: {strategy}")

    with st.expander("Add Custom Response Strategy"):
        custom_behavior = st.text_input("Behavior Name")
        custom_strategy = st.text_area("Response Strategy")
        if st.button("Add Strategy"):
            if custom_behavior and custom_strategy:
                RESPONSE_STRATEGIES[custom_behavior] = custom_strategy
                st.success(f"Added response strategy for {custom_behavior}")
            else:
                st.warning("Please provide both behavior name and strategy.")


# ---------------------- MODEL MANAGEMENT ---------------------- #
st.header("Model Management")
export_tab, = st.tabs(["Export Data"])

with export_tab:
    st.subheader("Export Data & Models")
    st.caption("Download data and models generated by the system for further analysis or backup.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Detection Data (CSV)")
        if os.path.exists("detection_data.csv"):
            with open("detection_data.csv", "rb") as f:
                st.download_button(
                    label="Download detection_data.csv",
                    data=f.read(),
                    file_name="detection_data.csv",
                    mime="text/csv"
                )
        else:
            st.info("No detection_data.csv found yet. Run an analysis to generate it.")

        st.markdown("### Tracking Data (CSV)")
        if os.path.exists("tracking_data.csv"):
            with open("tracking_data.csv", "rb") as f:
                st.download_button(
                    label="Download tracking_data.csv",
                    data=f.read(),
                    file_name="tracking_data.csv",
                    mime="text/csv"
                )
        else:
            st.info("No tracking_data.csv found yet. Run an analysis to generate it.")

    with col2:
        st.markdown("### LSTM Model (.h5)")
        if os.path.exists("lstm_crowd_behavior.h5"):
            with open("lstm_crowd_behavior.h5", "rb") as f:
                st.download_button(
                    label="Download LSTM Model",
                    data=f.read(),
                    file_name="lstm_crowd_behavior.h5",
                    mime="application/octet-stream"
                )
        else:
            st.info("LSTM model file not found in app directory.")

        st.markdown("### Q-Learning Data (.json)")
        if st.session_state.rl_agent.q_table:
            save_q_table_to_json("q_learning_data.json", st.session_state.rl_agent.q_table)

        if os.path.exists("q_learning_data.json"):
            with open("q_learning_data.json", "rb") as f:
                st.download_button(
                    label="Download Q-Learning Data",
                    data=f.read(),
                    file_name="q_learning_data.json",
                    mime="application/json"
                )
        else:
            st.info("No Q-learning data yet. Run analysis to update the RL agent.")


# ---------------------- DOCUMENTATION ---------------------- #
st.header("Documentation")
with st.expander("System Documentation"):
    st.markdown("""
    # AI-Powered Crowd Behavior Analysis System

    ## Features

    - Real-time people detection and tracking (YOLOv8 + ByteTrack)
    - Pose estimation and orientation analysis (YOLOv8-Pose)
    - Rule-based + LSTM-based behavior detection
    - ML-based aggression scoring (aggression_model.pkl)
    - Raised-hands pose heuristic for aggression
    - Heatmap visualization of crowd density and movement
    - Autonomous decision support (alert levels, emergency actions)
    - Q-learning driven adaptive thresholds
    - NLP-powered Chatbot Assistant for operators

    ## Behaviors Detected

    1. Calm – Normal behavior
    2. Dispersing – Moving away from central areas
    3. Aggressive – Erratic/hostile movements or raised-hands aggression
    4. Stampede – Fast, uniform flow indicating panic or rush

    ## Standard Response Protocol

    1. Maintain monitoring during Calm conditions
    2. Increase supervision during Dispersing
    3. Deploy security for Aggressive patterns
    4. Trigger emergency protocols for Stampede
    """)

st.markdown("---")
st.markdown("AI-Powered Crowd Behavior Analysis System - v1.0")

