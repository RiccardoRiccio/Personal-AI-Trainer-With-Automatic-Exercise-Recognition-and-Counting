#######################
## TRAIN THE FINAL DATASET (up till now we are using the dataset of new exercise without the synthetic or other dataset used before, so at some point concatanate and retrain)
######################

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from scipy.stats import randint, uniform
from sklearn.model_selection import ParameterSampler
import joblib
import matplotlib.pyplot as plt

# Load the preprocessed aggregated features and labels
# X_combined = np.load("X_similar_agg_blstm.npy")
# y_combined = np.load("y_similar_agg_blstm.npy")

# Load the preprocessed aggregated features and labels from multiple datasets
X_similar_agg = np.load("X_similar_dataset_newfeatures_landmarks_with_ids.npy")
y_similar_agg = np.load("y_similar_dataset_newfeatures_landmarks_with_ids.npy")

X_archive = np.load("X_final_thesis_finalkagglewithadditionalvideo_dataset_newfeatures_landmarks_with_ids.npy")
y_archive = np.load("y_final_thesis_finalkagglewithadditionalvideo_dataset_newfeatures_landmarks_with_ids.npy")

X_synthetic = np.load("X_synthetic_newfeatures_landmarks_with_ids.npy")
y_synthetic = np.load("y_synthetic_newfeatures_landmarks_with_ids.npy")

# Concatenate all datasets
X_combined = np.concatenate((X_similar_agg, X_archive, X_synthetic), axis=0)
y_combined = np.concatenate((y_similar_agg, y_archive, y_synthetic), axis=0)

# Validation checks
print("X_combined shape:", X_combined.shape)
print("y_combined shape:", y_combined.shape)

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_combined)
y_categorical = to_categorical(y_encoded)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_combined)


# Save scaler and label encoder
joblib.dump(scaler, 'thesis_lstm_scaler.pkl')
joblib.dump(label_encoder, 'thesis_lstm_label_encoder.pkl')

# Determine the number of timesteps and features
n_samples, n_features_total = X_scaled.shape
n_timesteps = 30  # Number of timesteps can be chosen based on the specific use case
n_features = n_features_total // n_timesteps

# Ensure the total number of features is divisible by the number of timesteps
if n_features_total % n_timesteps != 0:
    raise ValueError("The total number of features must be divisible by the number of timesteps.")

# Reshape data for LSTM
X_reshaped = X_scaled.reshape((n_samples, n_timesteps, n_features))

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y_categorical, test_size=0.2, random_state=42)

# Function to create LSTM model
def create_lstm_model(units=100, dropout_rate=0.5, learning_rate=0.001, regularizer=l2(0.01)):
    model = Sequential()
    model.add(LSTM(units, return_sequences=True, kernel_regularizer=regularizer, input_shape=(n_timesteps, n_features)))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(units, kernel_regularizer=regularizer))
    model.add(Dropout(dropout_rate))
    model.add(Dense(y_categorical.shape[1], activation='softmax', kernel_regularizer=regularizer))
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Hyperparameters distribution
param_dist = {
    'units': randint(50, 150),
    'dropout_rate': uniform(0.2, 0.5),
    'learning_rate': uniform(0.0001, 0.001),
    'batch_size': randint(32, 64),
    'epochs': randint(50, 100)
}

# Random search for hyperparameter tuning
n_iter_search = 20
random_search = list(ParameterSampler(param_dist, n_iter=n_iter_search, random_state=42))

best_model = None
best_accuracy = 0
best_params = None
best_history = None

for params in random_search:
    print(f"Testing with parameters: {params}")
    model = create_lstm_model(units=params['units'], dropout_rate=params['dropout_rate'], learning_rate=params['learning_rate'])
    
    # Implement early stopping and learning rate scheduling
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)

    history = model.fit(X_train, y_train, epochs=params['epochs'], batch_size=params['batch_size'], 
                        validation_data=(X_test, y_test), callbacks=[early_stopping, reduce_lr], verbose=1)
    
    # Evaluate the model
    y_pred_prob = model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = np.argmax(y_test, axis=1)
    accuracy = accuracy_score(y_true, y_pred)
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model
        best_params = params
        best_history = history

print(f"Best parameters: {best_params}")
print(f"Best accuracy: {best_accuracy:.2f}")

# Predict and evaluate
y_pred_prob = best_model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)
y_true = np.argmax(y_test, axis=1)

# Print classification report
print('Classification Report:')
print(classification_report(y_true, y_pred, target_names=label_encoder.classes_))

# Print confusion matrix
print('Confusion Matrix:')
print(confusion_matrix(y_true, y_pred))

# Save the best model
best_model.save('final_forthesis_lstmand_encoders_exercise_classifier_model.h5')

# Visualize learning curves
def plot_learning_curves(history):
    plt.figure(figsize=(12, 6))

    # Plot training & validation accuracy values
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.tight_layout()
    plt.show()

plot_learning_curves(best_history)