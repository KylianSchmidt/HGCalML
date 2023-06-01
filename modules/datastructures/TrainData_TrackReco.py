

from DeepJetCore.TrainData import TrainData, fileTimeOut
from DeepJetCore import SimpleArray
import numpy as np
import uproot
import awkward as ak

class TrainData_TrackReco(TrainData):
    def __init__(self):
        TrainData.__init__(self)
        # no class member is mandatory
        self.description = "This Class converts Root files from a custom Geant4 simulation to Awkward and Numpy arrays"
        #define any other (configuration) members that seem useful
        self.someusefulemember = "something you might need later"

        
    #def createWeighterObjects(self, allsourcefiles):
        # 
        # This function can be used to derive weights (or whatever quantity)
        # based on the entire data sample. It should return a dictionary that will then
        # be passed to either of the following functions. The weighter objects
        # should be pickleable.
        # In its default implementation, the dict is empty
        # return {}
        
        # This method extracts the events and hits for a given key in the root tree
    def extract_to_array(self, filename, key) :
        with uproot.open(filename) as data :
            return data[key].array(library="ak")
        
    # Construct the feature, truth (and weightobejects) from a root file
    # The data from the geant4 root files are jagged arrays of shape
    #   Properties x hits x eventNum
    # Where properties are listed here as keys. The prefix guides uproot to the correct branch->event
    def convertFromSourceFile(self, filename, weighterobjects, istraining):
        # Training Data
        prefix = "Events;1/Output/fHits/fHits."
        keys = ["layerType", "x", "y", "z", "E"]
        d, arr = {}, []
        for _, keys in enumerate(keys) :
            d[keys] = self.extract_to_array(filename=filename, key=prefix+keys)
            # Ignore the layerType == 0 hits as they record the hits in the absorber layer
            d[keys] = d[keys][d["layerType"] > 0]
            arr.append(d[keys][..., np.newaxis])
        feature_array = ak.concatenate(arr, axis=-1)

        # offsets for tensorflow have to be a np.array of dtype int
        offsets = np.cumsum(np.array(np.append(0, np.array(ak.to_list(ak.num(feature_array, axis=1))))))

        # Features are a np.array of dtype float32 and are rearranged into the shape
        #   eventNum x hits x properties
        # Where again hits is an index of variable length 
        feature_array = np.array(ak.to_list(ak.flatten(feature_array, axis=1)))
        feature_array = feature_array.astype(dtype='float32', order='C', casting="same_kind")

        # Truth data
        # In this case, the NN will be trained to recreate the vertex of the initial particles
        prefix = "Events;1/Output/fParticles/fParticles."
        keys = ["mcPx", "mcPy", "mcPz", "decayX", "decayY", "decayZ"]
        d, arr = {}, []
        for _, keys in enumerate(keys) :
            d[keys] = self.extract_to_array(filename=filename, key=prefix+keys)
            arr.append(d[keys][..., np.newaxis])
        truth_array = ak.concatenate(arr, axis=-1)
        # Transform into the desired shape (same as feature array) and slice the properties into 2 as we have
        # 2 photons in the root file
        truth_array = ak.flatten(ak.unflatten(ak.flatten(truth_array, axis=-1), counts=2, axis=-1), axis=0)
        truth_array = np.array(ak.to_list(truth_array)).astype(dtype='float32', order='C', casting="same_kind")

        self.nsamples = len(feature_array)
        
        print("Length of Features", self.nsamples)
        print("Length of Truths", len(truth_array))

        # Return a list of feature arrays, a list of truth arrays (and optionally a list of weight arrays)
        return [SimpleArray(feature_array, offsets, name="features0")], [truth_array], []
    
    ## defines how to write out the prediction
    def writeOutPrediction(self, predicted, features, truth, weights, outfilename, inputfile):
        # predicted will be a list of numpy arrays
        # save it as you like, the following way is not recommended as it is slow
        # and not disk-space efficient.
        # You can also use the fast and compressed TrainData format itself for saving
        # or use uproot to write a tree to a TFile
        
        with uproot.recreate(outfilename) as file :
            file["Predicted"] = predicted
            file["Features"] = features
            file["Weights"] = weights
            file["Truth"] = truth


        
