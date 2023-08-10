"""
Merges pose detection keypoint data with EEG data by timestamp.
"""

import glob
import pandas as pd



for file in glob.glob('sessions/EEG*.csv'):
	EEG_data = pd.read_csv(file, delimiter='\t')
	print(EEG_data['Timestamp'].head())
	EEG_data = EEG_data.drop(EEG_data.columns[[0, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 31]], axis=1)
	print(EEG_data['Timestamp'].head())
	EEG_data['Timestamp'] = EEG_data['Timestamp'].round(2)
	EEG_data.to_csv('sessions/EEG-test.csv', index=False)
	
for file in glob.glob('sessions/AIO*.csv'):
	Pose_data = pd.read_csv(file, delimiter=',')
	Pose_data['Timestamp'] = Pose_data['Timestamp'].round(2)
	print(Pose_data['Timestamp'].head())

merged_data = Pose_data.merge(EEG_data, how='inner', on='Timestamp')
merged_data.to_csv('prepared_data/Data.csv', sep=',', index=False)
