"""
Collects EEG data from OpenBCI.

See documentation for the API https://brainflow.readthedocs.io/en/stable/UserAPI.html#python-api-reference
"""

import time
import logging
import numpy as np
import pandas as pd

from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels, BoardIds
from brainflow.data_filter import DataFilter


BoardShim.enable_dev_board_logger()     #enable BrainFlow Logger with level TRACE, uses stderr for log messages by default
logging.basicConfig(level=logging.DEBUG)

params = BrainFlowInputParams()
#params.serial_port = 'COM3'

moment = time.strftime('%Y-%m-%d__%H-%M-%S', time.localtime())
eeg_channels = BoardShim.get_eeg_channels(BoardIds.SYNTHETIC_BOARD.value)       #get list of eeg channels in resulting data table for a board

try:
	board = BoardShim(BoardIds.SYNTHETIC_BOARD.value, params)    #Uses synth board for testing
	board.prepare_session()     #prepare streaming session, init resources, you need to call it before any other BoardShim object methods
	board.start_stream (450000)       #Start streaming data, this methods stores data in ringbuffer
	time.sleep(10)
	data = board.get_board_data()  # get all data and remove it from internal buffer
	DataFilter.write_file(data, 'sessions/EEG-'+moment+'.csv', 'a')
		
except BaseException:
	logging.warning('Exception', exc_info=True)

finally:
	logging.info('End')	
	if board.is_prepared():
		logging.info('Releasing session')
		board.release_session()     #release all resources
