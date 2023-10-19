'''
Training the AI to recognize poses from EEG.
Outputs both a keras and a tflite model.
'''

import pandas as pd
import numpy as np

np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf
from tensorflow.keras import layers

num_e = 1000 #number of epochs to train the AI

inputs = ['EXG Channel 0', 'EXG Channel 1', 'EXG Channel 2', 'EXG Channel 3', 'EXG Channel 4', 'EXG Channel 5', 'EXG Channel 6', 'EXG Channel 7', 'EXG Channel 8', 'EXG Channel 9', 'EXG Channel 10', 'EXG Channel 11', 'EXG Channel 12', 'EXG Channel 13', 'EXG Channel 14', 'EXG Channel 15']

outputs = ['nose y', 'nose x', 'left eye y', 'left eye x', 'right eye y', 'right eye x', 'left ear y', 'left ear x', 'right ear y', 'right ear x', 'left shoulder y', 'left shoulder x', 'right shoulder y', 'right shoulder x', 'left elbow y', 'left elbow x', 'right elbow y', 'right elbow x', 'left wrist y', 'left wrist x', 'right wrist y', 'right wrist x', 'left hip y', 'left hip x', 'right hip y', 'right hip x', 'left knee y', 'left knee x', 'right knee y', 'right knee x', 'left ankle y', 'left ankle x', 'right ankle y', 'right ankle x']

#Load the training data into a dataframe
training_dataset = pd.read_csv('prepared_data/Training_Data.csv') 
training_keypoints = training_dataset.copy()
training_labels = np.array(training_keypoints[outputs])
training_eeg = np.array(training_keypoints[inputs])

#Same thing for the validation data
validation_dataset = pd.read_csv('prepared_data/Validation_Data.csv')
validation_keypoints = validation_dataset.copy()
validation_labels = np.array(validation_keypoints[outputs])
validation_eeg = np.array(validation_keypoints[inputs])

#Train the model
pose_model = tf.keras.Sequential([layers.Dense(16), layers.Dense(64), layers.Dense(34)])
pose_model.compile(loss = tf.keras.losses.MeanSquaredError(), optimizer = tf.keras.optimizers.Adam())
pose_model.fit(training_eeg, training_labels, epochs=num_e, validation_data=(validation_eeg, validation_labels))

#Load test data
test_dataset = pd.read_csv('prepared_data/Test_Data.csv')
test_keypoints = test_dataset.copy()
test_labels = np.array(test_keypoints[outputs])
test_eeg = np.array(test_keypoints[inputs])

#Test the model with the test data
results = pose_model.evaluate(test_eeg, test_labels)
print('test loss & acc:', results)

#Saving the model
pose_model.save('models/keras/EEG_pose_R_622_'+str(num_e)+'_v2.keras')

#Convert to tflite and save
converter = tf.lite.TFLiteConverter.from_keras_model(pose_model)
tflite_model = converter.convert()
with open('models/tflite/EEG_pose_R_622_'+str(num_e)+'_v2.tflite', 'wb') as f:
  f.write(tflite_model)
