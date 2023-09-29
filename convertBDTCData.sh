#!/bin/bash
# Script wrapping convertFromSource but normalizes the input data for better
# performance. The normalization is kept in the same directory as the original

python3 normalize_inputs.py ./nntr_data/idealized_detector/Raw/Testing.root
python3 normalize_inputs.py ./nntr_data/idealized_detector/Raw/Training.root

python3 normalize_inputs.py ./nntr_data/normal_detector/Raw/Testing.root
python3 normalize_inputs.py ./nntr_data/normal_detector/Raw/Training.root

rm -rf ./nntr_data/idealized_detector/Testing
convertFromSource.py -i ./nntr_data/idealized_detector/Raw/test_files.txt \
                     -o ./nntr_data/idealized_detector/Testing \
                     -c TrainData_TrackReco

rm -rf ./nntr_data/idealized_detector/Training
convertFromSource.py -i ./nntr_data/idealized_detector/Raw/training_files.txt \
                     -o ./nntr_data/idealized_detector/Training \
                     -c TrainData_TrackReco

rm -rf ./nntr_data/normal_detector/Testing
convertFromSource.py -i ./nntr_data/normal_detector/Raw/test_files.txt \
                     -o ./nntr_data/normal_detector/Testing \
                     -c TrainData_TrackReco

rm -rf ./nntr_data/normal_detector/Training
convertFromSource.py -i ./nntr_data/normal_detector/Raw/training_files.txt \
                     -o ./nntr_data/normal_detector/Training \
                     -c TrainData_TrackReco