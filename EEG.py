"""
Collects EEG data from OpenBCI.

See documentation for the API https://brainflow.readthedocs.io/en/stable/UserAPI.html#python-api-reference

cyton daisy
--board-id 2
synthetic board
--board-id -1
"""

import argparse
import time
import csv

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter


parser = argparse.ArgumentParser()
parser.add_argument('--serial-port', type=str, help='serial port', required=False, default='/dev/ttyUSB0')
parser.add_argument('--board-id', type=int, help='board id, check docs to get a list of supported boards', required=False, default=BoardIds.CYTON_DAISY_BOARD) #Use BoardIds.SYNTHETIC_BOARD if you want to test with synthetic board
parser.add_argument('--master-board', type=int, help='master board id for streaming and playback boards', required=False, default=BoardIds.NO_BOARD)
args = parser.parse_args()

params = BrainFlowInputParams()
params.serial_port = args.serial_port
params.board_id = args.board_id
params.master_board = args.master_board


moment = time.strftime('%Y-%m-%d__%H-%M-%S', time.localtime())

with open('sessions/EEG-'+moment+'.csv', 'w+') as f:
	writer = csv.writer(f, delimiter='\t', dialect='excel')
	writer.writerow(['Sample Index', 'EXG Channel 0', 'EXG Channel 1', 'EXG Channel 2', 'EXG Channel 3', 'EXG Channel 4', 'EXG Channel 5', 'EXG Channel 6', 'EXG Channel 7', 'EXG Channel 8', 'EXG Channel 9', 'EXG Channel 10', 'EXG Channel 11', 'EXG Channel 12', 'EXG Channel 13', 'EXG Channel 14', 'EXG Channel 15', 'Accel Channel 0', 'Accel Channel 1', 'Accel Channel 2', 'Other', 'Other', 'Other', 'Other', 'Other', 'Other', 'Other', 'Analog Channel 0', 'Analog Channel 1', 'Analog Channel 2', 'Timestamp', 'Other'])
	f.close()

board = BoardShim(args.board_id, params)
board.prepare_session()           #Prepare streaming session, init resources, you need to call it before any other BoardShim object methods
board.start_stream ()       #Start streaming data, this method stores data in ring buffer

while True:
	try:               
        	time.sleep(4)
        	data = board.get_board_data()  #Get all data and remove it from internal buffer
        	DataFilter.write_file(data, 'sessions/EEG-'+moment+'.csv', 'a')
		
	except KeyboardInterrupt:
		data = board.get_board_data()
		DataFilter.write_file(data, 'sessions/EEG-'+moment+'.csv', 'a')
		board.release_session()     #Release all resources
		break
