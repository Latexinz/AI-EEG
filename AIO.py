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


interpreter = tf.lite.Interpreter(model_path='models/lite-model_movenet_singlepose_lightning_3.tflite')
interpreter.allocate_tensors()

# Timestamp for the filename.
moment = time.strftime('%Y-%m-%d__%H-%M-%S', time.localtime())	 

with open('sessions/AIO'+moment+'.csv', 'w+') as f:
	writer = csv.writer(f, dialect='excel')
	# Header for the file. Keypoints with x & y coordinates and models confidence
	writer.writerow(['nose x', 'nose y', 'nose c', 'left eye x', 'left eye y', 'left eye c', 'right eye x', 'right eye y', 'right eye c', 'left ear x', 'left ear y', 'left ear c', 'right ear x', 'right ear y', 'right ear c', 'left shoulder x', 'left shoulder y', 'left shoulder c', 'right shoulder x', 'right shoulder y', 'right shoulder c', 'left elbow x', 'left elbow y', 'left elbow c', 'right elbow x', 'right elbow y', 'right elbow c', 'left wrist x', 'left wrist y', 'left wrist c', 'right wrist x', 'right wrist y', 'right wrist c', 'left hip x', 'left hip y', 'left hip c', 'right hip x', 'right hip y', 'right hip c', 'left knee x', 'left knee y', 'left knee c', 'right knee x', 'right knee y', 'right knee c', 'left ankle x', 'left ankle y', 'left ankle c', 'right ankle x', 'right ankle y', 'right ankle c', 'Timestamp'])
	
	#video_path = 'videos/1.mp4'	# This is just for testing with videos.
	video_path = 0			
	cap = cv2.VideoCapture(video_path)
	cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')) # depends on fourcc available camera
	cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
	cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
	cap.set(cv2.CAP_PROP_FPS, 60)
	while cap.isOpened():
		ret, frame = cap.read()
		
		img = frame.copy()
		img = tf.image.resize_with_pad(np.expand_dims(img, axis=0), 192,192)
		input_image = tf.cast(img, dtype=tf.float32)
		
		input_details = interpreter.get_input_details()
		output_details = interpreter.get_output_details()
		
		interpreter.set_tensor(input_details[0]['index'], np.array(input_image))
		interpreter.invoke()
		keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])
		
		# Logging detections from frame
		keypoints_copy = keypoints_with_scores
		keypoints_list = keypoints_copy.flatten()
		keypoints_list.tolist()
		# Timestamp for the detection. Same format as OpenBCI.
		time_stamp = str(time.time() / 1000000000) + 'E9'
		keypoints_list = np.append(keypoints_list, time_stamp)
		writer.writerow(keypoints_list)
		
		# Calling functions to draw keypoints & connections on the frame
		draw_connections(frame, keypoints_with_scores, EDGES, 0.4)
		draw_keypoints(frame, keypoints_with_scores, 0.4)
		
		# Used for checking fps
		#fps = cap.get(cv2.CAP_PROP_FPS)
		#print(fps)
		
		cv2.imshow('MoveNet Lightning', frame)
		
		while cv2.waitKey(10) & 0xFF==ord('q'):
			break

	cap.release()
	cv2.destroyAllWindows()
