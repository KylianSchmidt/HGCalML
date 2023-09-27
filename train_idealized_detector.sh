#!/bin/bash

nnTrackReco_model_directory="./nntr_models/idealized_detector/${1}"

echo "Train the NN using the following model:"
echo $nnTrackReco_model_directory

# Train
echo "Commencing Training"
mkdir $nnTrackReco_model_directory
rm -rf $nnTrackReco_model_directory/Output
python3 Train/TrackReco_training.py \
        ./nntr_data/idealized_detector/Training/dataCollection.djcdc \
        $nnTrackReco_model_directory/Output \
        --valdata ./nntr_data/idealized_detector/Training/dataCollection.djcdc