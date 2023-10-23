#!/bin/bash

detector_type="idealized_detector"
model_name="v1_test"
take_weights=""

while getopts "d:m:w:" flag; do
    case "${flag}" in
        d) detector_type=$OPTARG;;
        m) model_name=$OPTARG;;
        w) take_weights="--takeweightsfrom $OPTARG";;
    esac
done

nntr_model_dir="./nntr_models/$detector_type/$model_name"

echo "Train the NN using the following model:"
echo $nntr_model_dir

# Train
echo "Starting Training"
SECONDS=0

mkdir $nntr_model_dir
rm -rf $nntr_model_dir/Output

python3 Train/nntr_training.py \
        ./nntr_data/$detector_type/Training/dataCollection.djcdc \
        $nntr_model_dir/Output \
        --valdata ./nntr_data/$detector_type/Training/dataCollection.djcdc
echo "Total training time: "$SECONDS

# Prediction
echo "Starting Prediction"


rm -rf $nntr_model_dir/Predicted
predict.py  $nntr_model_dir/Output/KERAS_check_best_model.h5 \
            $nntr_model_dir/Output/trainsamples.djcdc \
            ./nntr_data/$detector_type/Testing/Testing.djctd \
            $nntr_model_dir/Predicted



