from DeepJetCore import SimpleArray
import uproot
import numpy as np
from DeepJetCore import DataCollection


with uproot.open("nntr_data/normal_detector/Raw/Training_preprocessed.root") as file:
    hits_normalized = file["Hits"]["hits_normalized"].arrays(library="np")[
        "hits_normalized"]
    offsets = file["Hits_offsets_cumsum"]["offsets_cumsum"].arrays(library="np")[
        "offsets_cumsum"]
    truth_normalized = file["Truth"]["truth_normalized"].arrays(library="np")[
        "truth_normalized"]

    truth = truth_normalized.astype(
        dtype='float32',
        order='C',
        casting="same_kind")

    feature = hits_normalized.astype(
        dtype='float32',
        order='C',
        casting="same_kind")
    
arr = SimpleArray(feature, offsets, name="Features")

print(feature.shape)
print(arr.shape())
print("Truth", truth.shape)
print(truth)

train_data = DataCollection("/home/kschmidt/DeepJetCore/TrackReco_DeepJetCore/HGCalML/nntr_data/normal_detector/Training/dataCollection.djcdc")

traingen = train_data.invokeGenerator(fake_truth=False)

for event in traingen.feedTrainData():
    print(event)
