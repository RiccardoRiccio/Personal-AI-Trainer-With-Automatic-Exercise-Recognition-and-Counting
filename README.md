Fitness AI Coach: Automatic Exercise Recognition and Counting

This project is an AI-powered application that leverages Computer Vision, Pose Estimation, and Machine Learning to accurately track exercise repetitions during workouts. The goal is to enhance fitness routines by providing real-time feedback through an easy-to-use web interface.

Demo

Watch the Fitness AI Coach in action:



Table of Contents

Project Structure

Getting Started

Features

Technologies Used

Project Structure

main.py: Runs the Streamlit app.

ExerciseAiTrainer.py: Contains exercise-specific pose estimation logic.

AiTrainer_utils.py: Utility functions for image processing and distance calculations.

PoseModule2.py: Handles body pose estimation using MediaPipe.

chatbot.py: Implements the chatbot using the OpenAI API.

final_ldm_extractor.py: Extracts landmarks from videos.

final_preprocess.py: Preprocesses landmarks to generate features for model training.

final_train_lstm_and_encoders.py / final_train_bilstm_and_encoders.py: Scripts for training LSTM and BiLSTM models.

requirements.txt: List of required Python libraries.

shoulder_press_form.mp4: Sample video showing proper form for exercises.

thesis_bidirectionallstm_label_encoder.pkl / thesis_bidirectionallstm_scaler.pkl: Pre-trained models for label encoding and feature scaling.

Getting Started

Prerequisites

Python must be installed on your machine.

It's recommended to use a virtual environment to manage dependencies.

Installation

Clone this repository:

Install the required dependencies:

How to Run

To run the application, execute the following command in your terminal:

Features

Real-Time Exercise Tracking: Tracks repetitions for Bicep Curls, Push-Ups, Squats, and Shoulder Press using pose estimation and angle calculations.

Auto Classify Mode: Automatically classifies exercises in real-time using a BiLSTM model, eliminating the need for manual selection.

Manual Mode: Allows users to select specific exercises and count repetitions using an angle-based approach.

Chatbot Integration: Virtual personal trainer utilizing OpenAI to answer fitness-related questions.

Voice Commands: Users can start and stop exercise tracking via speech recognition.

Technologies Used

Pose Estimation: Utilizes MediaPipe to extract key body landmarks and monitor movement.

Machine Learning: LSTM and BiLSTM models for real-time exercise classification.

Speech Recognition: To enable voice-activated control of the application.

Streamlit: Web interface for interaction.

Python Libraries: OpenCV, MediaPipe, Streamlit, and more for backend processing.

Overview of the WebApp

The Fitness AI Coach is a web application built with Streamlit, aimed at providing users with tools for fitness tracking, real-time exercise classification, repetition counting, and chatbot support.

App Navigation

The main navigation sidebar allows users to access the following features:

Video Analysis: Users can upload exercise videos to count repetitions based on pose estimation.

Webcam Mode: Users can perform exercises in front of their webcam for real-time repetition counting.

Auto Classify Mode: Automatically identifies exercises in real-time and counts repetitions accordingly.

Chatbot: Acts as a fitness coach to provide fitness guidance using the OpenAI API.

The application is designed to be modular and user-friendly, with clear instructions and an intuitive interface for each feature. The system also provides visual cues and instructional videos to assist users with exercise form and repetition counts.

Implementation Details

Exercise Classifier

The exercise classifier is built using a mix of real and synthetic datasets to improve generalization. The main sources include:

Kaggle Workout Dataset: Real-world videos of various exercises.

InfiniteRep Dataset: Synthetic videos featuring avatars performing exercises to add diversity.

Similar Dataset: Videos from online sources to cover realistic variations.

The classification model employs LSTM and BiLSTM networks to process extracted landmarks and classify exercises based on joint angles and movement patterns. Hyperparameter tuning and evaluation were conducted using metrics like accuracy, precision, recall, and F1-score.

Repetition Counting

The repetition counting mechanism is implemented in two modes:

Manual Mode: Users manually select the exercise, and angle-based thresholds are applied to count repetitions.

Automatic Mode: A BiLSTM model classifies exercises, after which specific counting logic is applied based on identified body angles.

The counting logic relies on tracking "up" and "down" movements detected through angle variations, ensuring that the repetitions are accurately counted.

Chatbot Integration

The chatbot feature uses OpenAI's GPT-3.5-turbo model to answer user questions related to fitness and workouts. The chatbot is integrated into the web application using LangChain’s ConversationChain to maintain context and provide more meaningful responses.

A warning is displayed to inform users that the chatbot may occasionally provide incorrect information, and critical decisions should be verified with a professional.
