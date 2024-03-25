from plotting import Extract, FWHM

import pandas as pd
import awkward as ak
import numpy as np
import os, sys
import matplotlib.pyplot as plt
import scipy as sci
from icecream import ic


class ResultsTADistanceSweep():
    def __init__(self, ta_distance):
        self.ta_distance = ta_distance
        self.numFiles = range(20)
        self.data_tot = self.find_results_from_network()
        self.hits, self.truth, self.predicted = self.combine_results(self.data_tot)

    def find_results_from_network(self):
        model_name = f"tad_{self.ta_distance}"
        data_tot = []
        for i in self.numFiles:
            data_tot.append(
                Extract(
                    model_dir="nntr_models/normal_detector/ta_distance_sweep/",
                    model_name=model_name,
                    predicted_file=f"Predicted/pred_Testing_{i}_preprocessed.djctd",
                    testing_root_files=f"/ceph/kschmidt/beamdump/ta_distance_sweep/testing_{ta_distance}.0/Testing_{i}_preprocessed.root"
                    )
                )

        data = data_tot[0]
        self.savefigdir = ic(data.model_dir+data.model_name+"/Plots/")
        os.makedirs(self.savefigdir+"/Tracks/", exist_ok=True)
        return data_tot

    def combine_results(self, data_tot):
        """ Combine the results from the different files into one array """
        hits = []
        truth = []
        predicted = []
        for i in self.numFiles:
            hits.append(data_tot[i].hits)
            truth.append(data_tot[i].truth_array)
            predicted.append(data_tot[i].predicted_array)

        hits = ak.concatenate(hits, axis=0)
        truth = ak.concatenate(truth, axis=0)
        predicted = ak.concatenate(predicted, axis=0)
        return hits, truth, predicted

    def create_df(self):
        hits, truth, predicted = self.hits, self.truth, self.predicted
        df = pd.DataFrame({"Hits": hits.to_list(), "Truth": truth.to_list(), "Predicted": predicted.to_list()}, index=range(len(predicted)))
        df["VertexAccuracy"] = (truth[:, 14] - predicted[:, 14])
        df["V1z_true"] = truth[:, 14]
        df["V1z_pred"] = predicted[:, 14]
        df["E_tot"] = ak.sum(hits[:, :, -1], axis=1)
        df["AngleAccuracy"] = self.find_opening_angle(truth) - self.find_opening_angle(predicted)
        df["Theta_true"] = self.find_opening_angle(truth)
        df["RadialAccuracy"] = self.find_R(truth) - self.find_R(predicted)
        df["R_true"] = self.find_R(truth)
        return df
    
    def find_opening_angle(self, array):
        p1 = (array[:, 12:15] - array[:, 0:3]).to_numpy()
        p2 = (array[:, 15:18] - array[:, 6:9]).to_numpy()
        args = p1[:, 0]*p2[:, 0] + p1[:, 1]*p2[:, 1] + p1[:, 2]*p2[:, 2]
        theta = np.arccos(args/(np.linalg.norm(p1, axis=1)*np.linalg.norm(p2, axis=1)))
        #print(f"Opening angle = {theta} [rad] ({(theta*57.2958)}°)")
        return theta*57.2958
    
    def find_R(self, array):
        return np.sqrt(np.sum(array[:, 0:2].to_numpy()**2, axis=1))
    

detector_type = "normal_detector"
ta_distance = sys.argv[1]
results = ResultsTADistanceSweep(ta_distance)
hits, truth, predicted = results.hits, results.truth, results.predicted
df = results.create_df()
region_cut_off = -500
vertex_accuracy_near = df["VertexAccuracy"][df["V1z_true"] >= region_cut_off]
vertex_accuracy_far = df["VertexAccuracy"][df["V1z_true"] < region_cut_off]

ic(np.std(vertex_accuracy_near))
ic(np.std(vertex_accuracy_far))
vertex_accuracy_near.to_csv(f"/work/kschmidt/DeepJetCore/TrackReco_DeepJetCore/HGCalML/nntr_models/normal_detector/ta_distance_sweep/tad_{ta_distance}/Predicted/VertexAccuracyNear.csv")
vertex_accuracy_far.to_csv(f"/work/kschmidt/DeepJetCore/TrackReco_DeepJetCore/HGCalML/nntr_models/normal_detector/ta_distance_sweep/tad_{ta_distance}/Predicted/VertexAccuracyFar.csv")
df["VertexAccuracy"].to_csv(f"/work/kschmidt/DeepJetCore/TrackReco_DeepJetCore/HGCalML/nntr_models/normal_detector/ta_distance_sweep/tad_{ta_distance}/Predicted/VertexAccuracy.csv")

bin_width = 25
for bins in np.arange(-1400, -100, bin_width):
    vertex_accuracy_binned = df["VertexAccuracy"][(df["V1z_true"] > bins) & (df["V1z_true"] <= bins+bin_width)]
    vertex_accuracy_binned.to_csv(f"/work/kschmidt/DeepJetCore/TrackReco_DeepJetCore/HGCalML/nntr_models/normal_detector/ta_distance_sweep/tad_{ta_distance}/Predicted/VertexAccuracy_{bins}.csv")

    angular_accuracy_binned = df["AngleAccuracy"][(df["V1z_true"] > bins) & (df["V1z_true"] <= bins+bin_width)]
    angular_accuracy_binned.to_csv(f"/work/kschmidt/DeepJetCore/TrackReco_DeepJetCore/HGCalML/nntr_models/normal_detector/ta_distance_sweep/tad_{ta_distance}/Predicted/AngleAccuracy_{bins}.csv")

    radial_accuracy_binned = df["RadialAccuracy"][(df["V1z_true"] > bins) & (df["V1z_true"] <= bins+bin_width)]
    radial_accuracy_binned.to_csv(f"/work/kschmidt/DeepJetCore/TrackReco_DeepJetCore/HGCalML/nntr_models/normal_detector/ta_distance_sweep/tad_{ta_distance}/Predicted/RadialAccuracy_{bins}.csv")

bin_width = 5
for bins in np.arange(0, 100, bin_width):
    accuracy_binned = df["RadialAccuracy"][(df["R_true"] > bins) & (df["R_true"] <= bins+bin_width)]
    accuracy_binned.to_csv(f"/work/kschmidt/DeepJetCore/TrackReco_DeepJetCore/HGCalML/nntr_models/normal_detector/ta_distance_sweep/tad_{ta_distance}/Predicted/RadialAccuracyOverR_{bins}.csv")

bin_width = 4
for bins in np.arange(0, 60, bin_width):
    accuracy_binned = df["AngleAccuracy"][(df["Theta_true"] > bins) & (df["Theta_true"] <= bins+bin_width)]
    accuracy_binned.to_csv(f"/work/kschmidt/DeepJetCore/TrackReco_DeepJetCore/HGCalML/nntr_models/normal_detector/ta_distance_sweep/tad_{ta_distance}/Predicted/AngleAccuracyOverTheta_{bins}.csv")

