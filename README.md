# Personal AI Trainer With Automatic Exercise Recognition and Counting

This project is an AI-powered application that leverages Computer Vision, Pose Estimation, and Machine Learning to accurately track exercise repetitions during workouts. The goal is to enhance fitness routines by providing real-time feedback through an easy-to-use web interface.

## Demo

Watch the Fitness AI Coach in action:
[![Fitness AI Coach Gameplay Demo](https://img.youtube.com/vi/T-vpCzy17ik/0.jpg)](https://youtu.be/GPmDPB1bSmc "Personal AI Trainer With Automatic Exercise Recognition and Counting")

---

## Table of Contents
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the Application](#running-the-application)
- [Overview of the WebApp](#overview-of-the-webapp)
  - [App Navigation](#app-navigation)
- [Implementation Details](#implementation-details)
  - [Exercise Classifier](#exercise-classifier)
  - [Repetition Counting](#repetition-counting)
  - [Chatbot Integration](#chatbot-integration)
- [Technologies Used](#technologies-used)

---

## Project Structure
- `main.py`: Runs the Streamlit app.
- `ExerciseAiTrainer.py`: Contains exercise-specific pose estimation logic.
- `AiTrainer_utils.py`: Utility functions for image processing and distance calculations.
- `PoseModule2.py`: Handles body pose estimation using MediaPipe.
- `chatbot.py`: Implements the chatbot using the OpenAI API.
- `final_ldm_extractor.py`: Extracts landmarks from videos.
- `final_preprocess.py`: Preprocesses landmarks to generate features for model training.
- `final_train_lstm_and_encoders.py` / `final_train_bilstm_and_encoders.py`: Scripts for training LSTM and BiLSTM models.
- `requirements.txt`: List of required Python libraries.
- `shoulder_press_form.mp4`: Sample video showing proper form for exercises.
- `thesis_bidirectionallstm_label_encoder.pkl` / `thesis_bidirectionallstm_scaler.pkl`: Pre-trained models for label encoding and feature scaling.

---

## Getting Started

### Prerequisites
- Python 3.7+ must be installed on your machine.
- It's recommended to use a virtual environment to manage dependencies.

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/FitnessAI-Coach.git
   cd FitnessAI-Coach

2. Set up a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

3. Install the required Python libraries:

   ```bash
   pip install -r requirements.txt

## Overview of the WebApp

The Fitness AI Coach is a web application built with Streamlit, aimed at providing users with tools for fitness tracking, real-time exercise classification, repetition counting, and chatbot support.

### App Navigation

The main navigation sidebar allows users to access the following features:

1. **Video Analysis**: Upload exercise videos to count repetitions based on pose estimation.
2. **Webcam Mode**: Perform exercises in front of a webcam for real-time repetition counting.
3. **Auto Classify Mode**: Automatically identifies exercises in real-time and counts repetitions accordingly.
4. **Chatbot**: Acts as a fitness coach to provide fitness guidance using the OpenAI API.

The application is designed to be modular and user-friendly, with visual cues and instructional videos to assist users with exercise form and repetition counts.

## Implementation Details

### Exercise Classifier

The exercise classifier is built using a combination of real and synthetic datasets, including:

- **Kaggle Workout Dataset**: Real-world exercise videos.
- **InfiniteRep Dataset**: Synthetic videos of avatars performing exercises.
- **Similar Dataset**: Videos sourced from online to cover diverse exercise variations.

The classification model employs LSTM and BiLSTM networks to process body landmarks and classify exercises based on joint angles and movement patterns. The model was optimized using accuracy, precision, recall, and F1-score metrics.

### Repetition Counting

Repetition counting is implemented in two modes:

1. **Manual Mode**: Users manually select the exercise, and repetitions are counted using angle-based thresholds.
2. **Automatic Mode**: A BiLSTM model classifies exercises and applies counting logic based on identified body angles. The system tracks "up" and "down" movements to ensure accurate repetition counting.

### Chatbot Integration

The chatbot feature utilizes OpenAI's GPT-3.5-turbo model to answer fitness-related questions. It is integrated using LangChain’s ConversationChain to maintain context and provide meaningful responses. Users are advised to verify critical information with professionals as the chatbot may occasionally provide incorrect information.


## Technologies Used

- **Pose Estimation**: Utilizes MediaPipe to extract key body landmarks and monitor movement.
- **Machine Learning**: LSTM and BiLSTM models for real-time exercise classification.
- **Streamlit**: Provides the web interface for user interaction.
- **Python Libraries**: Includes OpenCV, MediaPipe, Streamlit, and others for backend processing.
