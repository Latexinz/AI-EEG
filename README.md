# AI-EEG

Tested on Ubuntu 20.05

Hardware used:
	-OpenBCI Ultracortex 'Mark IV' EEG headset with cyton daisy board
	-Logitech C922 webcam


Install necessary python libraries by running 'sh install.sh' in terminal.

Check for serial port permissions: Type 'id -Gn <username>' in terminal. If it prints dialout, great! Otherwise add the user to 'dialout'
with 'sudo usermod -a -G dialout <username>' and restart.

Run the python scripts to record data with 'bash run.sh' in terminal.

After that run Data_merge.py to merge the two recordings into a single file.
