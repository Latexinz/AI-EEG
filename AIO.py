'''
Model used can be found here: https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/3

This program runs a pose detection AI model on webcam video and records each keypoints position and detection confidence.

Camera used is Logitech C922, which supports 60fps @ 1280x720
'''

import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
import csv
import time
from os import cpu_count


def draw_keypoints(frame, keypoints, confidence_threshold):
	y, x, c = frame.shape
	shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))
	
	for kp in shaped:
		ky, kx, kp_conf = kp
		if kp_conf > confidence_threshold:
			cv2.circle(frame, (int(kx), int(ky)), 4, (0,255,0), -1) 


EDGES = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}


def draw_connections(frame, keypoints, edges, confidence_threshold):
	y, x, c = frame.shape
	shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))
	
	for edge, color in edges.items():
		p1, p2 = edge
		y1, x1, c1 = shaped[p1]
		y2, x2, c2 = shaped[p2]
        
		if (c1 > confidence_threshold) & (c2 > confidence_threshold):      
			cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 2)


# Use half the CPU for the interpreter, leave the rest for other stuff.
n_cores = int(cpu_count() / 2)
if cpu_count() > 3:
	print('Number of CPU threads assigned to the interpreter: ', n_cores)
else:
	print('Number of CPU threads assigned to the interpreter: 1')

interpreter = tf.lite.Interpreter(model_path='models/lite-model_movenet_singlepose_lightning_3.tflite', num_threads=n_cores)
interpreter.allocate_tensors()

# Timestamp for the filename.
moment = time.strftime('%Y-%m-%d__%H-%M-%S', time.localtime())	 

with open('sessions/AIO-'+moment+'.csv', 'w+') as f:
	writer = csv.writer(f, dialect='excel')
	# Header for the file. Keypoints with x & y coordinates and models confidence
	writer.writerow(['nose y', 'nose x', 'nose c', 'left eye y', 'left eye x', 'left eye c', 'right eye y', 'right eye x', 'right eye c', 'left ear y', 'left ear x', 'left ear c', 'right ear y', 'right ear x', 'right ear c', 'left shoulder y', 'left shoulder x', 'left shoulder c', 'right shoulder y', 'right shoulder x', 'right shoulder c', 'left elbow y', 'left elbow x', 'left elbow c', 'right elbow y', 'right elbow x', 'right elbow c', 'left wrist y', 'left wrist x', 'left wrist c', 'right wrist y', 'right wrist x', 'right wrist c', 'left hip y', 'left hip x', 'left hip c', 'right hip y', 'right hip x', 'right hip c', 'left knee y', 'left knee x', 'left knee c', 'right knee y', 'right knee x', 'right knee c', 'left ankle y', 'left ankle x', 'left ankle c', 'right ankle y', 'right ankle x', 'right ankle c', 'Timestamp'])
	

	video_path = 0		# Change this to 'videos/1.mp4' if testing without webcam	
	cap = cv2.VideoCapture(video_path)
	cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')) # Depends on fourcc available camera
	cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
	cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
	cap.set(cv2.CAP_PROP_FPS, 60)
	
	prev_frame_time = 0
	new_frame_time = 0
	
	while cap.isOpened():
		ret, frame = cap.read()
		
		# Crop & resize image
		frame = frame[0:720,280:1000]
		img = frame.copy()
		img = tf.image.resize_with_pad(np.expand_dims(img, axis=0), 192,192)
		input_image = tf.cast(img, dtype=tf.float32)
		
		input_details = interpreter.get_input_details()
		output_details = interpreter.get_output_details()
		
		# Make predictions
		interpreter.set_tensor(input_details[0]['index'], np.array(input_image))
		interpreter.invoke()
		keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])		
		
		# Turning the keypoints array into a list
		keypoints_copy = keypoints_with_scores
		keypoints_list = keypoints_copy.flatten()
		keypoints_list.tolist()
		# Timestamp for the detections. Same format as OpenBCI.
		time_stamp = str(time.time() / 1000000000) + 'E9'
		keypoints_list = np.append(keypoints_list, time_stamp)
		# Logging the keypoints into the csv file
		writer.writerow(keypoints_list)
		
		# Calling functions to draw keypoints & connections on the frame
		draw_connections(frame, keypoints_with_scores, EDGES, 0.4)
		draw_keypoints(frame, keypoints_with_scores, 0.4)
		
		# Use to check camera is outputting 60fps
		#fps = cap.get(cv2.CAP_PROP_FPS)
		#print(fps)
		
		# Displying detection fps
		font = cv2.FONT_HERSHEY_SIMPLEX
		new_frame_time = time.time()
		fps = 1/(new_frame_time-prev_frame_time)
		prev_frame_time = new_frame_time
		fps = str(int(fps))
		cv2.putText(frame, fps, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)
		
		cv2.imshow('MoveNet Lightning', frame)
		
		while cv2.waitKey(1) & 0xFF==ord('q'):
			break

	cap.release()
	cv2.destroyAllWindows()
