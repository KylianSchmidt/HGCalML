#!/bin/bash

nnTrackReco_model_directory="./nntr_models/${1}"

echo "Predict features based on the following model"
echo $nnTrackReco_model_directory

# Predict
echo "Commencing Prediction"

rm -rf $nnTrackReco_model_directory/Predicted
predict.py  $nnTrackReco_model_directory/Output/KERAS_check_best_model.h5 \
            $nnTrackReco_model_directory/Output/trainsamples.djcdc \
            ./nntr_data/PerfectDetectorTesting/Testing.djctd \
            $nnTrackReco_model_directory/Predicted


