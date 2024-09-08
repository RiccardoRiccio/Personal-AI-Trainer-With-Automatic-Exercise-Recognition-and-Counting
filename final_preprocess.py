#############################
### PREPROCESS LANDMARK, USE ADDITIONAL FEATURES LIKE NORMALIZED DISTANCES
###############################
import pandas as pd
import numpy as np

print("Preprocessing is running")

# Load the data
df = pd.read_csv('final_thesis_finaltestgym_dataset_newfeatures_landmarks_with_ids.csv')

# Define the expected number of landmarks (12 landmarks with x, y, z coordinates)
expected_landmark_count = 12

# Function to calculate angle between three points
def calculate_angle(a, b, c):
    if np.any(np.array([a, b, c]) == 0):
        return -1.0  # Placeholder for missing landmarks
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

# Function to calculate Euclidean distance between two points
def calculate_distance(a, b):
    if np.any(np.array([a, b]) == 0):
        return -1.0  # Placeholder for missing landmarks
    a = np.array(a)
    b = np.array(b)
    return np.linalg.norm(a - b)

# Function to calculate Y-coordinate distance between two points
def calculate_y_distance(a, b):
    if np.any(np.array([a, b]) == 0):
        return -1.0  # Placeholder for missing landmarks
    return np.abs(a[1] - b[1])

def extract_features(landmarks):
    features = []
    if len(landmarks) == expected_landmark_count:
        # Angles
        features.append(calculate_angle(landmarks[0], landmarks[2], landmarks[4]))  # LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST
        features.append(calculate_angle(landmarks[1], landmarks[3], landmarks[5]))  # RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST
        features.append(calculate_angle(landmarks[6], landmarks[8], landmarks[10]))  # LEFT_HIP, LEFT_KNEE, LEFT_ANKLE
        features.append(calculate_angle(landmarks[7], landmarks[9], landmarks[11]))  # RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE
        features.append(calculate_angle(landmarks[0], landmarks[6], landmarks[8]))  # LEFT_SHOULDER, LEFT_HIP, LEFT_KNEE
        features.append(calculate_angle(landmarks[1], landmarks[7], landmarks[9]))  # RIGHT_SHOULDER, RIGHT_HIP, RIGHT_KNEE

        # New angles
        features.append(calculate_angle(landmarks[6], landmarks[0], landmarks[2]))  # LEFT_HIP, LEFT_SHOULDER, LEFT_ELBOW
        features.append(calculate_angle(landmarks[7], landmarks[1], landmarks[3]))  # RIGHT_HIP, RIGHT_SHOULDER, RIGHT_ELBOW

        # Distances
        distances = [
            calculate_distance(landmarks[0], landmarks[1]),  # LEFT_SHOULDER, RIGHT_SHOULDER
            calculate_distance(landmarks[6], landmarks[7]),  # LEFT_HIP, RIGHT_HIP
            calculate_distance(landmarks[6], landmarks[8]),  # LEFT_HIP, LEFT_KNEE
            calculate_distance(landmarks[7], landmarks[9]),  # RIGHT_HIP, RIGHT_KNEE
            calculate_distance(landmarks[0], landmarks[6]),  # LEFT_SHOULDER, LEFT_HIP
            calculate_distance(landmarks[1], landmarks[7]),  # RIGHT_SHOULDER, RIGHT_HIP
            calculate_distance(landmarks[2], landmarks[8]),  # LEFT_ELBOW, LEFT_KNEE
            calculate_distance(landmarks[3], landmarks[9]),  # RIGHT_ELBOW, RIGHT_KNEE
            calculate_distance(landmarks[4], landmarks[0]),  # LEFT_WRIST, LEFT_SHOULDER
            calculate_distance(landmarks[5], landmarks[1]),  # RIGHT_WRIST, RIGHT_SHOULDER
            calculate_distance(landmarks[4], landmarks[6]),  # LEFT_WRIST, LEFT_HIP
            calculate_distance(landmarks[5], landmarks[7])   # RIGHT_WRIST, RIGHT_HIP
        ]

        # Y-coordinate distances
        y_distances = [
            calculate_y_distance(landmarks[2], landmarks[0]),  # LEFT_ELBOW, LEFT_SHOULDER
            calculate_y_distance(landmarks[3], landmarks[1])   # RIGHT_ELBOW, RIGHT_SHOULDER
        ]

        # Normalization factor based on shoulder-hip or hip-knee distance
        normalization_factor = -1
        distances_to_check = [
            calculate_distance(landmarks[0], landmarks[6]),  # LEFT_SHOULDER, LEFT_HIP
            calculate_distance(landmarks[1], landmarks[7]),  # RIGHT_SHOULDER, RIGHT_HIP
            calculate_distance(landmarks[6], landmarks[8]),  # LEFT_HIP, LEFT_KNEE
            calculate_distance(landmarks[7], landmarks[9])   # RIGHT_HIP, RIGHT_KNEE
        ]

        for distance in distances_to_check:
            if distance > 0:
                normalization_factor = distance
                break
        
        if normalization_factor == -1:
            normalization_factor = 0.5  # Fallback normalization factor
        
        # Normalize distances
        normalized_distances = [d / normalization_factor if d != -1.0 else d for d in distances]
        normalized_y_distances = [d / normalization_factor if d != -1.0 else d for d in y_distances]

        # Combine features
        features.extend(normalized_distances)
        features.extend(normalized_y_distances)

    else:
        print(f"Insufficient landmarks: expected {expected_landmark_count}, got {len(landmarks)}")
        features = [-1.0] * 22  # Placeholder for missing landmarks
    return features

def preprocess_data(df):
    X, y, video_ids = [], [], []
    for index, row in df.iterrows():
        video_id = row[0]
        exercise_type = row[1]
        landmarks = np.array(row[2:]).reshape(-1, 3)
        print(f"Processing row {index}, landmarks shape: {landmarks.shape}")
        if landmarks.shape[0] != expected_landmark_count:
            # Adjust landmarks array to have the expected shape
            adjusted_landmarks = np.zeros((expected_landmark_count, 3))
            adjusted_landmarks[:landmarks.shape[0], :] = landmarks
            landmarks = adjusted_landmarks
        features = extract_features(landmarks)
        X.append(features)
        y.append(exercise_type)
        video_ids.append(video_id)
    return np.array(X), np.array(y), np.array(video_ids)

X, y, video_ids = preprocess_data(df)

def aggregate_features(X, y, video_ids, window_size=30):
    X_agg, y_agg = [], []
    unique_videos = np.unique(video_ids)
    
    for video_id in unique_videos:
        video_indices = np.where(video_ids == video_id)[0]
        num_complete_windows = len(video_indices) // window_size
        for i in range(num_complete_windows):
            indices_window = video_indices[i * window_size:(i + 1) * window_size]
            X_window = X[indices_window].flatten()
            y_window = y[indices_window[0]]  # Use the label of the first frame in the window
            X_agg.append(X_window)
            y_agg.append(y_window)
    
    return np.array(X_agg), np.array(y_agg)

X_agg, y_agg = aggregate_features(X, y, video_ids, window_size=30)

# Save aggregated features
np.save('X_final_thesis_finaltestgym_dataset_newfeatures_landmarks_with_ids.npy', X_agg)
np.save('y_final_thesis_finaltestgym_dataset_newfeatures_landmarks_with_ids.npy', y_agg)
