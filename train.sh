#!/bin/bash

convertion=""
while getopts ch flag
do
    case "${flag}" in
        c) # Convert the data
        echo "Converting raw data"
        rm -rf ./NNTrackReco_PreparedData
        convertFromSource.py -i ./NNTrackReco_RawData/train_files.txt \
							 -o ./NNTrackReco_PreparedData \
							 -c TrainData_TrackReco
		echo "Conversion successful";;
        h) # Help
        echo "train.sh : Train the network"
        echo "Flag information"
        echo "-c Convert the raw data from root files to ragged arrays"
        exit 1;;
    esac
done

# Train
echo "Commencing Training"
rm -rf ./NNTrackReco_Output
python3 Train/TrackReco_training.py \
        ./NNTrackReco_PreparedData/dataCollection.djcdc \
        ./NNTrackReco_Output \
        --valdata ./NNTrackReco_PreparedData/dataCollection.djcdc