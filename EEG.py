"""
Collects EEG data from OpenBCI.

See documentation for the API https://brainflow.readthedocs.io/en/stable/UserAPI.html#python-api-reference

lsusb to check usb dongle for serial port

cyton daisy
--board-id 2
"""

import argparse
import time
import csv

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter


parser = argparse.ArgumentParser() # use docs to check which parameters are required for specific board, e.g. for Cyton - set serial port
parser.add_argument('--ip-port', type=int, help='ip port', required=False, default=0)
parser.add_argument('--ip-protocol', type=int, help='ip protocol, check IpProtocolType enum', required=False, default=0)
parser.add_argument('--ip-address', type=str, help='ip address', required=False, default='')
parser.add_argument('--serial-port', type=str, help='serial port', required=False, default='')
parser.add_argument('--mac-address', type=str, help='mac address', required=False, default='')
parser.add_argument('--serial-number', type=str, help='serial number', required=False, default='')
parser.add_argument('--board-id', type=int, help='board id, check docs to get a list of supported boards', required=False, default=BoardIds.SYNTHETIC_BOARD)
parser.add_argument('--master-board', type=int, help='master board id for streaming and playback boards', required=False, default=BoardIds.NO_BOARD)
args = parser.parse_args()

params = BrainFlowInputParams()
params.ip_port = args.ip_port
params.ip_protocol = args.ip_protocol
params.ip_address = args.ip_address
params.serial_port = args.serial_port
params.mac_address = args.mac_address
params.serial_number = args.serial_number
params.board_id = args.board_id
params.master_board = args.master_board


moment = time.strftime('%Y-%m-%d__%H-%M-%S', time.localtime())

with open('sessions/EEG-'+moment+'.csv', 'w+') as f:
	writer = csv.writer(f, dialect='excel')
	writer.writerow(['Sample Index', 'EXG Channel 0', 'EXG Channel 1', 'EXG Channel 2', 'EXG Channel 3', 'EXG Channel 4', 'EXG Channel 5', 'EXG Channel 6', 'EXG Channel 7', 'EXG Channel 8', 'EXG Channel 9', 'EXG Channel 10', 'EXG Channel 11', 'EXG Channel 12', 'EXG Channel 13', 'EXG Channel 14', 'EXG Channel 15', 'Accel Channel 0', 'Accel Channel 1', 'Accel Channel 2', 'Other', 'Other', 'Other', 'Other', 'Other', 'Other', 'Other', 'Analog Channel 0', 'Analog Channel 1', 'Analog Channel 2', 'Timestamp', 'Other'])
	f.close()

board = BoardShim(args.board_id, params)    #Uses synth board as default for testing
board.prepare_session()           #prepare streaming session, init resources, you need to call it before any other BoardShim object methods
board.start_stream ()       #Start streaming data, this method stores data in ring buffer

while True:
	try:               
        	time.sleep(10)
        	data = board.get_board_data()  # get all data and remove it from internal buffer
        	DataFilter.write_file(data, 'sessions/EEG-'+moment+'.csv', 'a')
		
	except KeyboardInterrupt:
		data = board.get_board_data()
		DataFilter.write_file(data, 'sessions/EEG-'+moment+'.csv', 'a')
		board.release_session()     #release all resources
		break
