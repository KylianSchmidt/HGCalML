#singularity run --nv /cvmfs/unpacked.cern.ch/registry.hub.docker.com/cernml4reco/deepjetcore3:latest
#source env.sh


# Convert the data
rm -rf ./NNTrackReco_PreparedData
rm -rf ./NNTrackReco_Output
convertFromSource.py -i ./NNTrackReco_RawData/train_files.txt -o ./NNTrackReco_PreparedData -c TrainData_TrackReco
echo "Conversion successful"

# Train
echo "Commencing Training"
python3 Train/TrackReco_training.py ./NNTrackReco_PreparedData/dataCollection.djcdc ./NNTrackReco_Output --valdata ./NNTrackReco_PreparedData/dataCollection.djcdc