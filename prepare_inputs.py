import numpy as np
import awkward as ak
import pandas as pd
import uproot
import sys
import itertools
import matplotlib.pyplot as plt
from tabulate import tabulate
from icecream import ic


class PrepareInputs:
    def __init__(self, filename):
        self.filename = filename.rstrip('.root')
        write = True

        # Extract hits and truth from root file
        raw_array = self._find_hits()
        truth_array = self._find_truth()

        offsets_with_empty_events = ak.count(raw_array, axis=1)[:, 0]

        # Remove empty events
        offsets_cumsum_with_empty_events = np.cumsum(offsets_with_empty_events)
        keep_index = np.zeros(len(offsets_cumsum_with_empty_events)-1, dtype='bool')

        for i in range(1, len(offsets_cumsum_with_empty_events)):
            if offsets_cumsum_with_empty_events[i-1] != offsets_cumsum_with_empty_events[i]:
                keep_index[i-1] = True

        raw_array = raw_array[keep_index]
        truth_array = truth_array[keep_index]

        # Remove all absorber hits
        raw_array = raw_array[raw_array[:, :, 0] != 0]
        offsets_without_absorber_cumsum = np.cumsum(ak.count(raw_array, axis=1)[:, 0])
        keep_index = np.zeros(len(offsets_without_absorber_cumsum)-1, dtype='bool')

        for i in range(1, len(offsets_without_absorber_cumsum)):
            if offsets_without_absorber_cumsum[i-1] != offsets_without_absorber_cumsum[i]:
                keep_index[i-1] = True

        raw_array = raw_array[keep_index]
        truth_array = truth_array[keep_index]

        offsets = ak.count(raw_array, axis=1)[:, 0]
        hits_flattened = np.array(ak.to_list(ak.flatten(raw_array)))
        truth_array = np.array(truth_array.tolist())

        # Create dataFrame (it is faster to create the pd.MultiIndex by hand than using
        # ak.to_dafaframe)
        print("[2/3] Convert to DataFrame and sum over same cellIDs")
        i_0 = list(np.repeat(np.arange(len(offsets)), offsets))
        i_1 = list(itertools.chain.from_iterable(
            [(np.arange(i)) for i in ak.to_list(offsets)]))
        nindex = pd.MultiIndex.from_arrays(arrays=[i_0, i_1], names=["event", "hit"])

        # Add truth
        truth_c = i_0 = np.repeat(truth_array, offsets, axis=0)
        truth_keys = [
            "Px1", "Py1", "Pz1", "Vx1", "Vy1", "Vz1",
            "Px2", "Py2", "Pz2", "Vx2", "Vy2", "Vz2"
        ]

        df_original = pd.DataFrame(
            np.concatenate([hits_flattened, truth_c], axis=1),
            columns=["layerType", "cellID", "x", "y", "z", "E", *truth_keys],
            index=nindex)

        # Sum over same cellIDs in the calorimeter
        df = df_original.groupby(["event", "cellID"], dropna=False)[
            ["layerType", "x", "y", "z", *truth_keys]].first()
        df["E"] = df_original.groupby(["event", "cellID"], dropna=False)["E"].sum()

        # Convert momentum direction to aligned points in truth
        truth_array_grouped = df.groupby(["event"])[[*truth_keys]].first().to_numpy()

        def line(truth, particle_num, z=0):
            if particle_num == 1:
                p = truth[:, 0:3]
                v = truth[:, 3:6]
            elif particle_num == 2:
                p = truth[:, 6:9]
                v = truth[:, 9:12]
            x = p[:, 0]/p[:, 2]*(z - v[:, 2])
            y = p[:, 1]/p[:, 2]*(z - v[:, 2])
            z = np.repeat(z, len(p))
            return np.stack([x, y, z], axis=-1)

        A1 = line(truth_array_grouped, 1, 0)
        B1 = line(truth_array_grouped, 1, 300)
        A2 = line(truth_array_grouped, 2, 0)
        B2 = line(truth_array_grouped, 2, 300)
        V1 = truth_array_grouped[:, 3:6]
        V2 = truth_array_grouped[:, 9:12]

        truth_points = np.concatenate([A1, B1, A2, B2, V1, V2], axis=1)
        assert len(truth_points) == len(truth_array_grouped)

        # Remove events with low energy
        offsets_sum_calo = self._find_offsets(df)
        features_sum_calo = df[["layerType", "x", "y", "z", "E"]].to_numpy()
        features = ak.unflatten(features_sum_calo, offsets_sum_calo)

        mask_lower_E_cut = ak.sum(features[:, :, 4], axis=1) > 0.2
        features = features[mask_lower_E_cut]
        truth_points = truth_points[mask_lower_E_cut.to_numpy()]
        truth_array_grouped = truth_array_grouped[mask_lower_E_cut.to_numpy()]
        num_after_low_E_cut = len(ak.flatten(features))

        # Remove events with small opening angle
        p1 = truth_points[:, 12:15] - truth_points[:, 0:3]
        p2 = truth_points[:, 15:18] - truth_points[:, 6:9]
        args = p1[:, 0]*p2[:, 0] + p1[:, 1]*p2[:, 1] + p1[:, 2]*p2[:, 2]
        theta_arr = np.arccos(args/(np.linalg.norm(p1, axis=1)*np.linalg.norm(p2, axis=1)))*57.2958
        mask_theta = theta_arr > 5
        features = features[mask_theta]
        truth_points = truth_points[mask_theta]
        truth_array_grouped = truth_array_grouped[mask_theta]
        num_after_angle_cut = len(ak.flatten(features))

        # Remove events with low number of hits
        mask_low_hit_num = ak.count(features, axis=1)[:, 0] > 20
        features = features[mask_low_hit_num]
        truth_points = truth_points[mask_low_hit_num]
        truth_array_grouped = truth_array_grouped[mask_low_hit_num]
        num_after_low_hit_cut = len(ak.flatten(features))

        # Record offsets
        offsets = ak.count(features, axis=1)[:, 0]
        offsets_cumsum = np.append(0, np.cumsum(offsets))
        features = np.array(ak.to_list(ak.flatten(features)))

        # Normalized truth and features
        truth_mean = np.mean(truth_points, axis=0)+1E-10
        truth_std = np.std(truth_points, axis=0)+1E-10
        truth_points_normalized = (truth_points-truth_mean)/truth_std

        mean = np.mean(features, axis=0)+1E-10
        std = np.std(features, axis=0)+1E-10
        hits_normalized = (features-mean)/std



        # Check that everything is correct
        assert len(offsets_cumsum) == len(truth_points)+1
        totlen = offsets_cumsum_with_empty_events[-1]
        print(tabulate(
            [["Original root file:", totlen, 100],
             ["Removed absorber hits:", offsets_without_absorber_cumsum[-1], f"{offsets_without_absorber_cumsum[-1]/totlen*100:.2f}"],
             ["Summed over calo cells:", len(features_sum_calo), f"{len(features_sum_calo)/totlen*100:.2f}"],
             ["Removed low E", num_after_low_E_cut, f"{num_after_low_E_cut/totlen*100:.2f}"],
             ["Removed small opening angle", num_after_angle_cut, f"{num_after_angle_cut/totlen*100:.2f}"],
             ["Removed low hit count", num_after_low_hit_cut, f"{num_after_low_hit_cut/totlen*100:.2f}"]],
            headers=["", "Number of events", "Ratio [%]"]))

        # Write to file
        if write:
            with uproot.recreate(f"{self.filename}_preprocessed.root") as file:
                file["Hits"] = {"hits_normalized": hits_normalized}
                file["Hits_row_splits"] = {"rowsplits": offsets_cumsum}
                file["Hits_offsets"] = {"offsets": offsets}
                file["Hits_parameters"] = {"hits_mean": mean, "hits_std": std}
                file["Truth"] = {"truth_normalized": truth_points_normalized}
                file["Truth_parameters"] = {"truth_mean": truth_mean, "truth_std": truth_std}

        # Perfect detector
        perfect_hits = []
        for x in np.concatenate([np.linspace(0, 250, 30), [400]]):
            perfect_hits.append(line(truth_array_grouped, 1, x))
            perfect_hits.append(line(truth_array_grouped, 2, x))
        perfect_hits = np.transpose(np.array(perfect_hits), axes=(1, 0, 2))
        perfect_hits = np.concatenate(
            [np.full((perfect_hits.shape[0], perfect_hits.shape[1], 1), 1),
             perfect_hits,
             np.full((perfect_hits.shape[0], perfect_hits.shape[1], 1), 0.0002)],
            axis=2)
        perfect_hits[:, -1, 0] = 2.0
        perfect_hits[:, -1, 4] = 0.04
        perfect_hits[:, -2, 0] = 2.0
        perfect_hits[:, -2, 4] = 0.04
        rng = np.random.default_rng()
        rng.shuffle(perfect_hits, axis=1)

        # Record offsets
        perfect_offsets = ak.count(perfect_hits, axis=1)[:, 0]
        perfect_offsets_cumsum = np.append(0, np.cumsum(perfect_offsets))
        perfect_hits = np.array(ak.to_list(ak.flatten(perfect_hits)))

        # Normalize
        perfect_hits_mean = np.mean(perfect_hits, axis=0)+1E-10
        perfect_hits_std = np.std(perfect_hits, axis=0)+1E-10
        perfect_hits_normalized = (perfect_hits-perfect_hits_mean)/perfect_hits_std

        assert len(perfect_offsets_cumsum) == len(truth_points)+1

        # Write to file
        if write:
            with uproot.recreate(f'{self.filename.replace("normal", "idealized")}_preprocessed.root') as file:
                file["Hits"] = {"hits_normalized": perfect_hits_normalized}
                file["Hits_row_splits"] = {"rowsplits": perfect_offsets_cumsum}
                file["Hits_offsets"] = {"offsets": perfect_offsets}
                file["Hits_parameters"] = {"hits_mean": perfect_hits_mean, "hits_std": perfect_hits_std}
                file["Truth"] = {"truth_normalized": truth_points_normalized}
                file["Truth_parameters"] = {"truth_mean": truth_mean, "truth_std": truth_std}

    def _find_hits(self) -> ak.Array:
        keys = ["layerType", "cellID", "x", "y", "z", "E"]
        prefix = "Events/Output/fHits/fHits."
        raw_array = []

        print("[1/3] Extract from root file")
        for keys in keys:
            raw_array.append(self._extract_to_array(
                prefix+keys)[..., np.newaxis])
        raw_array = ak.concatenate(raw_array, axis=-1)
        return raw_array

    def _find_truth(self) -> ak.Array:
        """ Shape:
        (events x 12)
        """
        prefix = "Events/Output/fParticles/fParticles."
        keys = ["mcPx", "mcPy", "mcPz", "decayX", "decayY", "decayZ"]
        truth_array = []

        for key in keys:
            truth_array.append(self._extract_to_array(
                prefix+key)[..., np.newaxis])
        truth_array = ak.concatenate(truth_array, axis=-1)
        truth_array = ak.flatten(truth_array, axis=-1)
        return truth_array

    def _extract_to_array(self, key) -> ak.Array:
        """ This method extracts the events and hits for a given key in the root
        tree

        Returns shapes:
        (hits x [layerType, x, y, z, E])
        """
        with uproot.open(self.filename+".root") as data:
            return data[key].array(library="ak")

    def _find_offsets(self, df: pd.DataFrame) -> np.ndarray:
        return df.reset_index(level=1).index.value_counts().sort_index().to_numpy()


if __name__ == "__main__":
    PrepareInputs("./nntr_data/normal_detector/Raw/Testing.root")
    PrepareInputs("./nntr_data/normal_detector/Raw/Training.root")

