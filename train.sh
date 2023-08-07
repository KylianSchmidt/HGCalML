#!/bin/bash

nnTrackReco_model_directory="./NNTrackReco/${1}"

echo "Train the NN using the following model:"
echo $nnTrackReco_model_directory

# Train
echo "Commencing Training"
mkdir $nnTrackReco_model_directory
rm -rf $nnTrackReco_model_directory/Output
python3 Train/TrackReco_training.py \
        ./NNTrackReco_data/PreparedData/dataCollection.djcdc \
        $nnTrackReco_model_directory/Output \
        --valdata ./NNTrackReco_data/PreparedData/dataCollection.djcdc