'''
Removes last ~10s of recordings
Filters the EEG data.
Merges pose detection keypoint data with EEG data by timestamp.
'''

import glob
import pandas as pd
import numpy as np

from Filters import butter_highpass_filter, butter_bandpass_filter, Implement_Notch_Filter



EEG_data = pd.concat([pd.read_table(file, sep='\t') for file in glob.glob('sessions/EEG*.csv')])        #all sessions into a single dataframe
EEG_data = EEG_data.drop(EEG_data.columns[[0, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 31]], axis=1)
EEG_data['Timestamp'] = EEG_data['Timestamp'].round(2)
EEG_data = EEG_data[:-1250]
        
channel_num = 0
while channel_num < 16:                     #loop to go through each channel and apply filters to raw EEG signals
	channel_name = 'EXG Channel ' + str(channel_num)
	#plot raw here
	EEG_data[channel_name] = butter_highpass_filter(EEG_data[channel_name], 0.5, 250)
	EEG_data[channel_name] = butter_bandpass_filter(EEG_data[channel_name], 0.5, 40.0, 250.0, order=6)
	EEG_data[channel_name] = Implement_Notch_Filter(0.004, 1, 50, 1, 2, 'butter', EEG_data[channel_name])
	#plot filtered here
	channel_num = channel_num + 1


Pose_data = pd.concat([pd.read_csv(file, delimiter=',') for file in glob.glob('sessions/AIO*.csv')])
Pose_data['Timestamp'] = Pose_data['Timestamp'].round(2)
Pose_data = Pose_data[:-600]

merged_data = Pose_data.merge(EEG_data, how='inner', on='Timestamp') #merge the datasets
merged_data = merged_data.drop(['Timestamp'], axis=1)
trn_data, vld_data, tst_data = np.split(merged_data, [int(.6*len(merged_data)), int(.8*len(merged_data))]) #split the data into training, validation and testing data
trn_data.to_csv('prepared_data/Training_Data.csv', sep=',', index=False)
vld_data.to_csv('prepared_data/Validation_Data.csv', sep=',', index=False)
tst_data.to_csv('prepared_data/Test_Data.csv', sep=',', index=False)
