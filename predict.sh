#!/bin/bash

convertion=""
while getopts c flag
do
    case "${flag}" in
        c) # Convert the data
        echo "Converting raw data"
        rm -rf ./NNTrackReco_Testing/
        convertFromSource.py  -i ./NNTrackReco_RawData/test_files.txt \
                      -o ./NNTrackReco_Testing \
                      -c TrainData_TrackReco
		echo "Conversion successful";;
    esac
done


predict.py  ./NNTrackReco_Output/KERAS_model.h5 \
            ./NNTrackReco_Output/trainsamples.djcdc \
            ./NNTrackReco_Testing/Testing.djctd \
              NNTrackReco_Predicted


