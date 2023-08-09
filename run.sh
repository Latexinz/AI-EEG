#!/bin/bash

python3 EEG.py &
python3 AIO.py &
trap 'kill $(jobs -p)' EXIT
wait
