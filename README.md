# Fitness AI Application Code
This repository contains the code for the Fitness AI application, which is designed to provide real-time exercise classification, repetition counting, and interactive fitness guidance through a chatbot. The application is structured as a web app built with Streamlit and integrates several advanced AI and machine learning techniques, including pose estimation and LSTM models for exercise classification.

Overview
The Fitness AI application is divided into four main functionalities:

Video Analysis: Allows users to upload videos of their exercises, perform pose estimation using MediaPipe, and count repetitions based on detected movements.
Webcam Mode: Enables real-time exercise tracking and repetition counting through a webcam. The system uses similar pose estimation techniques as the video analysis feature.
Auto Classify Mode: Automatically detects the type of exercise being performed using a BiLSTM model and applies the appropriate repetition counting logic.
Chatbot: Acts as a fitness coach, providing users with personalized fitness advice and guidance based on their interactions. The chatbot is designed to behave as an expert fitness trainer and maintains conversational context to offer more tailored responses.
Code Structure
main.py: The main entry point for the Streamlit application. This script integrates all functionalities, including video analysis, webcam mode, auto classify mode, and the chatbot interface.
final_ldm_extractor.py: Handles the landmark extraction from exercise videos using MediaPipe for pose estimation.
final_preprocess.py: Processes extracted landmarks to create features suitable for model training, including angle calculations and normalized distances.
final_train_lstm_and_encoders.py and final_train_bilstm_and_encoders.py: Scripts for training the LSTM and BiLSTM models used in exercise classification.
chatbot.py: Manages the chatbot's interaction flow and integrates the OpenAI API for generating responses.
Important Note on Videos
This repository does not include instructional videos as they exceed GitHub's file size limits (25MB). As a result, certain functionalities, such as displaying instructional videos in the app, may not work directly with the provided code. To enable these features, you can uncomment specific lines in main.py related to video loading, and manually add your own videos to the project directory.
