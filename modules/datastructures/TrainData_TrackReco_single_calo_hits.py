from DeepJetCore.TrainData import TrainData
from DeepJetCore import SimpleArray
import numpy as np
import uproot
import awkward as ak
import pickle
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class TrainData_TrackReco(TrainData):
    def __init__(self):
        TrainData.__init__(self)
        self.description = "This Class converts Root files from a Geant4 " + \
                           "simulation to Awkward and Numpy arrays"
        
    def extract_to_array(self, key) -> ak.Array:
        """ This method extracts the events and hits for a given key in the root
        tree
        """
        with uproot.open(self.filename) as data:
            return data[key].array(library="ak")
        
    def find_hits(self) -> ak.Array:
        keys = ["layerType", "x", "y", "z", "E"]
        prefix = "Events/Output/fHits/fHits."
        d, arr = {}, []

        for keys in keys:
            d[keys] = self.extract_to_array(prefix+keys)
            d[keys] = d[keys].mask[d["layerType"] > 0]
            arr.append(d[keys][..., np.newaxis])

        return ak.concatenate(arr, axis=-1)
    
    def find_truth(self) -> ak.Array:
        """ In this case, the NN will be trained to recreate the vertex of the
        initial particles
        Also normalizes the inputs to have mean 0 and std 1
        """
        keys = ["mcPx", "mcPy", "mcPz", "decayX", "decayY", "decayZ"]
        prefix_truth = "Events/Output/fParticles/fParticles."
        arr = []

        for keys in keys:
            truth_original = self.extract_to_array(prefix_truth+keys)/1000
            self.truth_mean = np.mean(truth_original, axis=0, keepdims=True)
            self.truth_std = np.std(truth_original, axis=0, keepdims=True)
            truth_normalized = (truth_original - self.truth_mean)/(self.truth_std+1E-10)
            arr.append(truth_normalized[..., np.newaxis])

        truth_array = ak.concatenate(arr, axis=-1)
        truth_array = ak.flatten(truth_array, axis=-1)
        truth_array = np.array(ak.to_list(truth_array))

        self.truth_mean = np.mean(truth_array, axis=0)
        self.truth_std = np.std(truth_array, axis=0)
        truth_array = (truth_array - self.truth_mean)/(self.truth_std+1E-10)
        return truth_array
    
    def find_offsets(self, features) -> np.ndarray:
        return np.cumsum(
            np.append(
                0, np.array(
                    ak.to_list(
                        ak.num(features, axis=1)))))

    def convertFromSourceFile(
            self,
            filename,
            weightobjects,
            istraining):
        """ Construct the feature, truth (and weightobejects) from a root file
        The data from the geant4 root files are jagged arrays of shape

            Properties x hits x eventNum

        Where properties are listed here as keys. The prefix guides uproot
        to the correct branch->event

        Truth has the shape:
            0   1   2   3   4   5   6   7   8   9   10  11
           [px1 py1 pz1 vx1 vy1 vz1 px2 py2 pz2 vx2 vy2 vz2]
        """
        self.filename = filename

        feature_array = self.find_hits()
        offsets = self.find_offsets(feature_array)
        truth_array = self.find_truth()

        logger.info(
            "\nData Info:\n----------\n" +
            f"Feature array | (with empty Events) | {len(feature_array)}\n" +
            f"Truth array   | (with empty Events) | {len(truth_array)}")

        feature_array, truth_array = self.remove_empty_events(
            feature_array, truth_array, offsets)
        
        truth_array = truth_array.astype(
            dtype='float32',
            order='C',
            casting="same_kind")
        
        feature_array = ak.to_numpy(ak.flatten(feature_array, axis=1))
        feature_array = feature_array.astype(
            dtype='float32',
            order='C',
            casting="same_kind")
        offsets = np.unique(offsets)

        logging.info(
            f"Feature array| (without empty Events) | {np.shape(feature_array)}\n" +
            f"Truth array  | (without empty Events) | {np.shape(truth_array)}\n" +
            f"Offsets      | (without empty Events) | {np.shape(offsets)}\n" +
            "Mean and std of truth over N_events:\n" +
            f"mean = {self.truth_mean}\n" +
            f"std  = {self.truth_std}\n" +
            "----------")
        
        return ([SimpleArray(feature_array, offsets, name="Features")],
                [truth_array],
                [])

    def remove_empty_events(
            self,
            features: np.ndarray,
            truth: np.ndarray,
            offsets_cumulative: np.ndarray):
        """ Function which removes empty events by checking for double entries
        in the cumulative offsets arrays. This might be useful when layers
        compute objects like "means" which are ill-defined when there are empty
        arrays
        """
        keep_index = np.zeros(len(offsets_cumulative)-1, dtype='bool')

        for i in range(1, len(offsets_cumulative)):
            if offsets_cumulative[i-1] != offsets_cumulative[i]:
                keep_index[i-1] = True

        logging.info("Removed empty arrays")
        return features[keep_index], truth[keep_index]

    def writeOutPrediction(
            self,
            predicted,
            features,
            truth,
            weights,
            outfilename,
            inputfile) -> None:
        """ Defines the way the predicted data is stored to disk. Currently
        done using pickle.

        Predicted will be a list of numpy arrays
        """

        data = {
            "Hits": features,
            "Truth": truth[0],
            "Predicted": predicted[0][:, 0:12],
            "Uncertainties": predicted[0][:, 12:24]}

        with open(outfilename, "wb") as file:
            pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)
        return None
