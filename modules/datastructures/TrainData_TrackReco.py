

from DeepJetCore.TrainData import TrainData, fileTimeOut
from DeepJetCore import SimpleArray
import numpy as np
import uproot
import awkward as ak
import pickle

class TrainData_TrackReco(TrainData):
    def __init__(self):
        TrainData.__init__(self)
        # no class member is mandatory
        self.description = "This Class converts Root files from a Geant4 simulation to Awkward and Numpy arrays"
        #define any other (configuration) members that seem useful
        self.someusefulemember = "something you might need later"
        
    def extract_to_array(self, filename, key) :
        """This method extracts the events and hits for a given key in the root tree"""
        with uproot.open(filename) as data :
            return data[key].array(library="ak")
        
    def convertFromSourceFile(self, filename, weightobjects, istraining):
        """ Construct the feature, truth (and weightobejects) from a root file
        The data from the geant4 root files are jagged arrays of shape
            Properties x hits x eventNum
        Where properties are listed here as keys. The prefix guides uproot to the correct branch->event\n
        Features have the shape:\n
        Truth array has the shape:\n
            0   1   2   3   4   5   6   7   8   9   10  11  12  13
           [px1 py1 pz1 n1  vx1 vy1 vz1 px2 py2 pz2 n2  vx2 vy2 vz2]
        """
        # Training Data
        keys = ["layerType", "x", "y", "z", "E"]
        prefix = f"Events/Output/fHits/fHits."
        d, arr = {}, []
        for _, keys in enumerate(keys) :
            d[keys] = self.extract_to_array(filename=filename, key=prefix+keys)
            # Ignore the "layerType == 0" hits as they record the hits in the absorber layer
            d[keys] = d[keys][d["layerType"] > 0]
            arr.append(d[keys][..., np.newaxis])

        feature_array = ak.concatenate(arr, axis=-1)
        # Offsets for TensorFlow have to be a "np.array" of "dtype int"
        offsets = np.cumsum(np.array(np.append(0, np.array(ak.to_list(ak.num(feature_array, axis=1))))))

        # Truth data
        # In this case, the NN will be trained to recreate the vertex of the initial particles
        keys = ["mcPx", "mcPy", "mcPz", "decayX", "decayY", "decayZ"]
        prefix = f"Events/Output/fParticles/fParticles."
        arr = []
        for _, keys in enumerate(keys) :
            d = self.extract_to_array(filename=filename, key=prefix+keys)
            arr.append(d[..., np.newaxis])
        truth_array = ak.concatenate(arr, axis=-1)
        truth_array = ak.flatten(truth_array, axis=-1)
        
        print("Feature array | (with empty Events) |", len(feature_array))
        print("Truth array   | (with empty Events) |", len(truth_array))
        print("Truth array[0] =", truth_array[0])

        # Deletes empty events which can be due to both photons missing the detector
        feature_array, truth_array = self.remove_empty_events(feature_array, truth_array, offsets)

        feature_array = ak.to_numpy(ak.flatten(feature_array, axis=1))
        feature_array = feature_array.astype(dtype='float32', order='C', casting="same_kind")

        norm_p1 = np.sqrt(np.sum(truth_array[:,0:3]**2, axis=-1))
        norm_p2 = np.sqrt(np.sum(truth_array[:,7:10]**2, axis=-1))
        
        truth_array = np.insert(truth_array, 3, norm_p1, axis=-1)
        truth_array = np.insert(truth_array, 10, norm_p2, axis=-1)
        truth_array = np.array(ak.to_list(truth_array))
        truth_array[:,0:3] = truth_array[:,0:3] / (np.array(ak.to_list(norm_p1+1E-10))[..., np.newaxis])
        truth_array[:,7:10] = truth_array[:,7:10] / (np.array(ak.to_list(norm_p2+1E-10))[..., np.newaxis])
        truth_array = truth_array.astype(dtype='float32', order='C', casting="same_kind")

        offsets = np.unique(offsets)

        print("Feature array | (without empty Events) |", np.shape(feature_array))
        print("Truth array   | (without empty Events) |", np.shape(truth_array))
        print("Offsets       | (without empty Events) |", np.shape(offsets))

        # Return a list of feature arrays, a list of truth arrays (and optionally a list of weight arrays)
        return [SimpleArray(feature_array, offsets, name="Features")], [truth_array], []

    def remove_empty_events(self, features, truth, offsets_cumulative) :
        """ Function which removes empty events by checking for double entries
        in the cumulative offsets arrays. This might be useful when layers compute
        objects like "means" which are ill-defined when there are empty arrays
        """
        keep_index = np.zeros(len(offsets_cumulative)-1, dtype='bool')
        for i in range(1, len(offsets_cumulative)) :
            if offsets_cumulative[i-1] != offsets_cumulative[i] :
                keep_index[i-1] = True
        print("Removed empty arrays")
        return features[keep_index], truth[keep_index]

    def writeOutPrediction(self, predicted, features, truth, weights, outfilename, inputfile):
        """ Defines the way the predicted data is stored to disk. Currently done
        using pickle. 
        """
        # predicted will be a list of numpy arrays
        data = {"Features" : features,
                "Truth" : truth,
                "Predicted" : predicted}
        
        print(outfilename)
        with open(outfilename, "wb") as file:
            pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)
        
