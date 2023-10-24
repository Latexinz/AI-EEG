'''
Test

synthetic board
--board-id -1
'''

import argparse
import tensorflow as tf
import numpy as np
import pandas as pd
import time
import cv2

from Filters import butter_highpass_filter, butter_bandpass_filter, Implement_Notch_Filter
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds



def draw_keypoints(frame, keypoints):
	y, x, c = frame.shape
	shaped = np.squeeze(np.multiply(keypoints, [y,x]))
	
	for kp in shaped:
		ky, kx = kp
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


def draw_connections(frame, keypoints, edges):
	y, x, c = frame.shape
	shaped = np.squeeze(np.multiply(keypoints, [y,x]))
	
	for edge, color in edges.items():
		p1, p2 = edge
		y1, x1 = shaped[p1]
		y2, x2 = shaped[p2]
             
		cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 2)

#This function offsets the keypoints to the center of the drawn image
def adjust_keypoints(keypoints):
	kp = np.squeeze(keypoints)
	y, x = kp[0]  
        
	y_offset = y - 0.25
	x_offset = x - 0.5

	kp_new = [[y - y_offset, x - x_offset] for y,x in kp ]
	kp_new = np.around(np.array(kp_new),3)
	return kp_new


parser = argparse.ArgumentParser()
parser.add_argument('--serial-port', type=str, help='serial port', required=False, default='/dev/ttyUSB0')
parser.add_argument('--board-id', type=int, help='board id, check docs to get a list of supported boards', required=False, default=BoardIds.CYTON_DAISY_BOARD)
args = parser.parse_args()

params = BrainFlowInputParams()
params.serial_port = args.serial_port
params.board_id = args.board_id

board = BoardShim(args.board_id, params)
board.prepare_session()
board.start_stream()
time.sleep(1)

interpreter = tf.lite.Interpreter(model_path='models/tflite/EEG_pose_R_622_500_v2.tflite')
interpreter.allocate_tensors()

prev_frame_time = 0
new_frame_time = 0

while True:
	try:
		
		data = board.get_board_data()           
		data = pd.DataFrame(np.transpose(data[1:17]), index=None, columns=['EXG Channel 0', 'EXG Channel 1', 'EXG Channel 2', 'EXG Channel 3', 'EXG Channel 4', 'EXG Channel 5', 'EXG Channel 6', 'EXG Channel 7', 'EXG Channel 8', 'EXG Channel 9', 'EXG Channel 10', 'EXG Channel 11', 'EXG Channel 12', 'EXG Channel 13', 'EXG Channel 14', 'EXG Channel 15'])
            
		channel_num = 0
		while channel_num < 16:
			channel_name = 'EXG Channel ' + str(channel_num)
			data[channel_name] = butter_highpass_filter(data[channel_name], 0.5, 125)
			data[channel_name] = butter_bandpass_filter(data[channel_name], 0.5, 40.0, 125.0, order=6)
			data[channel_name] = Implement_Notch_Filter(0.004, 1, 50, 1, 2, 'butter', data[channel_name])
			channel_num = channel_num + 1
		
		data = np.array(data)
		i = 0
		
		while i < len(data):
			#time.sleep(0.1)
			img = np.zeros((960, 960, 3), np.uint8)
			
			input_data = tf.convert_to_tensor(data[i], dtype=tf.float32)
			input_data = np.reshape(input_data, (1, 16))
			i = i + 1
			
			input_details = interpreter.get_input_details()			
			output_details = interpreter.get_output_details()

			interpreter.set_tensor(input_details[0]['index'], input_data)
			interpreter.invoke()
			keypoints = interpreter.get_tensor(output_details[0]['index'])

			keypoints = np.reshape(keypoints, (17, 2))
			keypoints = adjust_keypoints(keypoints)

			draw_connections(img, keypoints, EDGES)
			draw_keypoints(img, keypoints)

			font = cv2.FONT_HERSHEY_SIMPLEX
			new_frame_time = time.time()
			fps = 1/(new_frame_time-prev_frame_time)
			prev_frame_time = new_frame_time
			fps = str(int(fps))
			cv2.putText(img, fps, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)

			cv2.imshow('EEG pose demo', img)

			while cv2.waitKey(1) & 0xFF==ord('q'):
				break
                
	except KeyboardInterrupt:
		break
