'''
Removes last ~10s of recordings
Filters the EEG data.
Merges pose detection keypoint data with EEG data by timestamp.
'''

import glob
import pandas as pd

from Filters import butter_highpass_filter, butter_bandpass_filter, Implement_Notch_Filter


for file in glob.glob('sessions/EEG*.csv'):
    EEG_data = pd.read_table(file, sep='\t')    #read as table since BrainFlow records data as tsv      
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

for file in glob.glob('sessions/AIO*.csv'):
	Pose_data = pd.read_csv(file, delimiter=',')
	Pose_data['Timestamp'] = Pose_data['Timestamp'].round(2)
	Pose_data = Pose_data[:-600]

merged_data = Pose_data.merge(EEG_data, how='inner', on='Timestamp')
#merged_data = merged_data.drop(merged_data.columns[['Timestamp']], axis=1)
merged_data.to_csv('prepared_data/Data.csv', sep=',', index=False)

