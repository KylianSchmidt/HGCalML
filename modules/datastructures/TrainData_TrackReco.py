from DeepJetCore.TrainData import TrainData
from DeepJetCore import SimpleArray
import numpy as np
import uproot
import awkward as ak
import pickle
import os


class TrainData_TrackReco(TrainData):
    def __init__(self):
        TrainData.__init__(self)
        self.description = "This Class converts Root files from a Geant4 " + \
                           "simulation to Awkward and Numpy arrays"
        
    def extract_to_array(self, filename, key):
        """This method extracts the events and hits for a given key in the root
        tree"""
        with uproot.open(filename) as data:
            return data[key].array(library="ak")
        
    def find_hits(self, filename):
        keys = ["layerType", "x", "y", "z", "E"]
        prefix = "Events/Output/fHits/fHits."
        d, arr = {}, []

        for keys in keys:
            d[keys] = self.extract_to_array(filename, prefix+keys)
            d[keys] = d[keys].mask[d["layerType"] > 0]
            arr.append(d[keys][..., np.newaxis])

        return ak.concatenate(arr, axis=-1)
    
    def find_truth(self, filename):
        """ In this case, the NN will be trained to recreate the vertex of the
        initial particles
        """
        keys = ["mcPx", "mcPy", "mcPz", "decayX", "decayY", "decayZ"]
        prefix = "Events/Output/fParticles/fParticles."
        arr = []

        for keys in keys:
            d = self.extract_to_array(filename, prefix+keys)/1000
            arr.append(d[..., np.newaxis])

        truth_array = ak.concatenate(arr, axis=-1)
        truth_array = ak.flatten(truth_array, axis=-1)
        return truth_array
    
    def find_offsets(self, features):
        return np.cumsum(
                    np.append(
                        0, np.array(
                            ak.to_list(
                                ak.num(features, axis=1)))))
    
    def normalization(self,
                      filename: str,
                      mode="read") -> None:
        """ This method saves the normalization parameters to disk
        Hardcoded path for now since the actual path is managed by
        DeepJetCore internally.
        """
        # NOTE: I would rather use Typing.Literal for 'mode' but DJC
        # uses python 3.6.9

        file = os.path.basename(os.path.normpath(filename))
        file = file.rstrip("root").rstrip(".")
        file = os.path.join("/home/kschmidt/DeepJetCore/" +
                            "TrackReco_DeepJetCore/HGCalML/" +
                            "nntr_data/normalization", f"{file}.pkl")
        print("Normalization file:", file)

        if mode == "write":
            with open(file, "w+b") as file:
                pickle.dump({"Mean": self.truth_mean,
                            "Std": self.truth_std}, file)
        elif mode == "read":
            with open(file, "r+b") as file:
                data = pickle.load(file)
                self.truth_mean = data["Mean"]
                self.truth_std = data["Std"]

        return None

    def convertFromSourceFile(self, filename, weightobjects, istraining):
        """ Construct the feature, truth (and weightobejects) from a root file
        The data from the geant4 root files are jagged arrays of shape

            Properties x hits x eventNum

        Where properties are listed here as keys. The prefix guides uproot
        to the correct branch->event

        Truth has the shape:
            0   1   2   3   4   5   6   7   8   9   10  11
           [px1 py1 pz1 vx1 vy1 vz1 px2 py2 pz2 vx2 vy2 vz2]
        """

        feature_array = self.find_hits(filename)
        offsets = self.find_offsets(feature_array)
        truth_array = self.find_truth(filename)

        print("Feature array | (with empty Events) |", len(feature_array))
        print("Truth array   | (with empty Events) |", len(truth_array))

        feature_array, truth_array = self.remove_empty_events(
                                        feature_array, truth_array, offsets)

        truth_array = np.array(ak.to_list(truth_array))
        self.truth_mean = np.mean(truth_array, axis=0)
        self.truth_std = np.std(truth_array, axis=0)
        truth_array = (truth_array - self.truth_mean)/(self.truth_std+1E-10)
        
        self.normalization(filename, mode="write")

        feature_array = ak.to_numpy(ak.flatten(feature_array, axis=1))
        feature_array = feature_array.astype(dtype='float32',
                                             order='C',
                                             casting="same_kind")
        truth_array = truth_array.astype(dtype='float32',
                                         order='C',
                                         casting="same_kind")

        offsets = np.unique(offsets)

        self.print_info(truth_array, feature_array, offsets)
        return ([SimpleArray(feature_array, offsets, name="Features")],
                [truth_array],
                [])

    def remove_empty_events(self, features, truth, offsets_cumulative):
        """ Function which removes empty events by checking for double entries
        in the cumulative offsets arrays. This might be useful when layers
        compute objects like "means" which are ill-defined when there are empty
        arrays
        """
        keep_index = np.zeros(len(offsets_cumulative)-1, dtype='bool')

        for i in range(1, len(offsets_cumulative)):
            if offsets_cumulative[i-1] != offsets_cumulative[i]:
                keep_index[i-1] = True

        print("Removed empty arrays")
        return features[keep_index], truth[keep_index]

    def writeOutPrediction(self,
                           predicted,
                           features,
                           truth,
                           weights,
                           outfilename,
                           inputfile):
        """ Defines the way the predicted data is stored to disk. Currently
        done using pickle.

        Predicted will be a list of numpy arrays
        """
        self.normalization(inputfile, mode="read")
        truth = truth[0]*(self.truth_std+1E-10) + self.truth_mean
        predicted = predicted[0][:, 0:12]*(self.truth_std+1E-10) \
            + self.truth_mean
        uncertainties = predicted[0][:, 12:24]*(self.truth_std+1E-10) \
            + self.truth_mean

        data = {"Hits": features,
                "Truth": truth,
                "Predicted": predicted,
                "Uncertainties": uncertainties,
                "Mean": self.truth_mean,
                "Std": self.truth_std}

        with open(outfilename, "wb") as file:
            pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)

    def print_info(self, truth_array, feature_array, offsets):
        print(f"Feature array|(without empty Events)|{np.shape(feature_array)}")
        print(f"Truth array  |(without empty Events)|{np.shape(truth_array)}")
        print(f"Offsets      |(without empty Events)|{np.shape(offsets)}")
        print("Mean and std of truth over N_events:")
        print(f"mean = {self.truth_mean}")
        print(f"std  = {self.truth_std}")        
