#!/bin/bash

nnTrackReco_model_directory="./NNTrackReco/${1}"

echo "Predict features based on the following model"
echo $nnTrackReco_model_directory

# Predict
echo "Commencing Prediction"

rm -rf $nnTrackReco_model_directory/Predicted
predict.py  $nnTrackReco_model_directory/Output/KERAS_model.h5 \
            $nnTrackReco_model_directory/Output/trainsamples.djcdc \
            ./NNTrackReco_data/TestingData/Testing.djctd \
            $nnTrackReco_model_directory/Predicted


