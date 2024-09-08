####################################
### EXTRACT LANDMARK FROM VIDEO, HERE JUST ONE SIDE, LEFT OR RIGHT IS ESSENTISAL TO MAINTAIN THE FRAME
###################################
import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
import os

# Initialize MediaPipe pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Define relevant landmarks indices
relevant_landmarks_indices = [
    mp_pose.PoseLandmark.LEFT_SHOULDER.value,
    mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
    mp_pose.PoseLandmark.LEFT_ELBOW.value,
    mp_pose.PoseLandmark.RIGHT_ELBOW.value,
    mp_pose.PoseLandmark.LEFT_WRIST.value,
    mp_pose.PoseLandmark.RIGHT_WRIST.value,
    mp_pose.PoseLandmark.LEFT_HIP.value,
    mp_pose.PoseLandmark.RIGHT_HIP.value,
    mp_pose.PoseLandmark.LEFT_KNEE.value,
    mp_pose.PoseLandmark.RIGHT_KNEE.value,
    mp_pose.PoseLandmark.LEFT_ANKLE.value,
    mp_pose.PoseLandmark.RIGHT_ANKLE.value
]

# Essential landmarks for each exercise
essential_landmarks = {
    'barbell biceps curl': [mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
                            mp_pose.PoseLandmark.LEFT_ELBOW.value, mp_pose.PoseLandmark.RIGHT_ELBOW.value,
                            mp_pose.PoseLandmark.LEFT_WRIST.value, mp_pose.PoseLandmark.RIGHT_WRIST.value],
    'push-up': [mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
                mp_pose.PoseLandmark.LEFT_ELBOW.value, mp_pose.PoseLandmark.RIGHT_ELBOW.value,
                mp_pose.PoseLandmark.LEFT_WRIST.value, mp_pose.PoseLandmark.RIGHT_WRIST.value,
                mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.RIGHT_HIP.value],
    'squat': [mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.RIGHT_HIP.value,
              mp_pose.PoseLandmark.LEFT_KNEE.value, mp_pose.PoseLandmark.RIGHT_KNEE.value,
              mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
    'shoulder press': [mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
                       mp_pose.PoseLandmark.LEFT_ELBOW.value, mp_pose.PoseLandmark.RIGHT_ELBOW.value,
                       mp_pose.PoseLandmark.LEFT_WRIST.value, mp_pose.PoseLandmark.RIGHT_WRIST.value]
}

# Function to extract landmarks from a video
def extract_landmarks(video_path, exercise_type, video_id):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return []
    
    landmarks = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame_landmarks = []
        if results.pose_landmarks:
            left_side_valid = True
            right_side_valid = True
            for idx in relevant_landmarks_indices:
                if idx < len(results.pose_landmarks.landmark):
                    landmark = results.pose_landmarks.landmark[idx]
                    frame_landmarks.extend([landmark.x, landmark.y, landmark.z])
                else:
                    frame_landmarks.extend([0.0, 0.0, 0.0])
                    if idx in essential_landmarks[exercise_type]:
                        if idx % 2 == 0:  # Left side indices are even
                            left_side_valid = False
                        else:  # Right side indices are odd
                            right_side_valid = False
            
            if left_side_valid or right_side_valid:
                landmarks.append([video_id, exercise_type] + frame_landmarks)
            else:
                print(f"Skipping frame {frame_count} due to missing critical landmarks.")
        else:
            print(f"No landmarks detected in frame {frame_count}. Skipping this frame.")

    cap.release()
    print(f"Extracted {len(landmarks)} valid frames from {frame_count} total frames in {video_path}")
    return landmarks

def main():
    data_dir = 'final_test_gym'
    exercise_types = ['barbell biceps curl', 'push-up', 'squat', 'shoulder press']
    data = []

    for exercise in exercise_types:
        exercise_dir = os.path.join(data_dir, exercise)
        if not os.path.isdir(exercise_dir):
            print(f"Error: {exercise_dir} is not a directory")
            continue
        video_files = [f for f in os.listdir(exercise_dir) if f.endswith('.mp4')]
        for video_id, video_file in enumerate(video_files):
            try:
                video_path = os.path.join(exercise_dir, video_file)
                landmarks = extract_landmarks(video_path, exercise, video_id)
                if landmarks:
                    data.extend(landmarks)
                else:
                    print(f"No landmarks detected for {video_file}")
            except Exception as e:
                print(f"Error processing {video_file}: {e}")

    if data:
        df = pd.DataFrame(data)
        output_file = 'final_thesis_finaltestgym_dataset_newfeatures_landmarks_with_ids.csv'
        df.to_csv(output_file, index=False)
        print(f"Data saved to {output_file}")
    else:
        print("No data to save.")

if __name__ == "__main__":
    main()
