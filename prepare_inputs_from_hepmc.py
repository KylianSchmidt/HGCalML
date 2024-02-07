import numpy as np
import awkward as ak
import pandas as pd
import uproot
import os
import itertools
import matplotlib.pyplot as plt
from tabulate import tabulate
from icecream import ic
from multiprocessing import Pool
plt.style.use("belle2")


class PrepareInputs:
    def __init__(self):
        self.truth_mean = None
        self.truth_std = None
        self.hits_mean = None
        self.hits_std = None
        self.perfect_hits_mean = None
        self.perfect_hits_std = None

    def perform(
            self,
            filename,
            energy_cut=0.2,
            low_hits_cut=20,
            opening_angle_cut=2,
            perfect_detector=False,
            write=True
            ):
        self.filename = ic(filename).rstrip('.root')
        self.hits_checkpoint_filename = self.filename+"_df_checkpoint.csv"
        self.truth_checkpoint_filename = self.filename+"_truth_checkpoint.csv"
        truth_keys = [
                "Px1", "Py1", "Pz1", "Vx1", "Vy1", "Vz1",
                "Px2", "Py2", "Pz2", "Vx2", "Vy2", "Vz2"
            ]
        offsets_with_empty_ = np.array([1])
        offsets_without_absorber_cumsum = np.array([1])
        offsets_cumsum_with_empty_ = np.array([1])

        if os.path.exists(self.hits_checkpoint_filename):
            df = pd.read_csv(self.hits_checkpoint_filename, index_col=[0, 1])
        else:
            # Extract hits and truth from root file
            raw_array = self._find_hits()
            truth = self._find_truth()

            offsets_with_empty_ = ak.count(raw_array, axis=1)[:, 0]

            # Remove empty 
            offsets_cumsum_with_empty_ = np.cumsum(offsets_with_empty_)
            keep_index = np.zeros(len(offsets_cumsum_with_empty_)-1, dtype='bool')

            for i in range(1, len(offsets_cumsum_with_empty_)):
                if offsets_cumsum_with_empty_[i-1] != offsets_cumsum_with_empty_[i]:
                    keep_index[i-1] = True

            raw_array = raw_array[keep_index]
            truth = truth[keep_index]

            # Remove all absorber hits
            raw_array = raw_array[raw_array[:, :, 0] != 0]
            offsets_without_absorber_cumsum = np.cumsum(ak.count(raw_array, axis=1)[:, 0])
            keep_index = np.zeros(len(offsets_without_absorber_cumsum)-1, dtype='bool')

            for i in range(1, len(offsets_without_absorber_cumsum)):
                if offsets_without_absorber_cumsum[i-1] != offsets_without_absorber_cumsum[i]:
                    keep_index[i-1] = True

            raw_array = raw_array[keep_index]
            truth = truth[keep_index]

            offsets = ak.count(raw_array, axis=1)[:, 0]
            hits_flattened = np.array(ak.to_list(ak.flatten(raw_array)))
            truth = np.array(truth.tolist())

            # Create dataFrame (it is faster to create the pd.MultiIndex by hand than using
            # ak.to_dafaframe)
            print("[2/3] Convert to DataFrame and sum over same cellIDs")
            i_0 = list(np.repeat(np.arange(len(offsets)), offsets))
            i_1 = list(itertools.chain.from_iterable(
                [(np.arange(i)) for i in ak.to_list(offsets)]))
            nindex = pd.MultiIndex.from_arrays(
                arrays=[i_0, i_1], names=["event", "hit"])

            # Add truth
            truth_c = i_0 = np.repeat(truth, offsets, axis=0)

            df_original = pd.DataFrame(
                np.concatenate([hits_flattened, truth_c], axis=1),
                columns=["layerType", "cellID", "x",
                         "y", "z", "E", *truth_keys],
                index=nindex)

            # Sum over same cellIDs in the calorimeter
            df = df_original.groupby(["event", "cellID"], dropna=False)[["layerType", "x", "y", "z", *truth_keys]].first()
            df["E"] = df_original.groupby(["event", "cellID"], dropna=False)["E"].sum()

            # Checkpoint
            df.to_csv(self.hits_checkpoint_filename)

        # Convert momentum direction to aligned points in truth
        truth_grouped = df.groupby(["event"])[[*truth_keys]].first().to_numpy()

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

        A1 = line(truth_grouped, 1, 0)
        B1 = line(truth_grouped, 1, 300)
        A2 = line(truth_grouped, 2, 0)
        B2 = line(truth_grouped, 2, 300)
        V1 = truth_grouped[:, 3:6]
        V2 = truth_grouped[:, 9:12]

        truth_points = np.concatenate([A1, B1, A2, B2, V1, V2], axis=1)
        assert len(truth_points) == len(truth_grouped)

        # Remove event with low energy
        offsets_sum_calo = self._find_offsets(df)
        features_sum_calo = df[["layerType", "x", "y", "z", "E"]].to_numpy()
        features = ak.unflatten(features_sum_calo, offsets_sum_calo)
        mask_lower_E_cut = ak.sum(features[:, :, 4], axis=1) > energy_cut
        features = features[mask_lower_E_cut]
        truth_points = truth_points[mask_lower_E_cut.to_numpy()]
        truth_grouped = truth_grouped[mask_lower_E_cut.to_numpy()]
        num_after_low_E_cut = len(ak.flatten(features))

        # Remove event with small opening angle
        p1 = truth_points[:, 12:15] - truth_points[:, 0:3]
        p2 = truth_points[:, 15:18] - truth_points[:, 6:9]
        args = p1[:, 0]*p2[:, 0] + p1[:, 1]*p2[:, 1] + p1[:, 2]*p2[:, 2]
        theta_arr = np.arccos(args/(np.linalg.norm(p1, axis=1)*np.linalg.norm(p2, axis=1)))*57.2958
        mask_theta = theta_arr > opening_angle_cut
        features = features[mask_theta]
        truth_points = truth_points[mask_theta]
        truth_grouped = truth_grouped[mask_theta]
        num_after_angle_cut_ = len(features)
        num_after_angle_cut_hits = len(ak.flatten(features))

        # Remove events with low number of hits
        mask_low_hit_num = ak.count(features, axis=1)[:, 0] > low_hits_cut
        features = features[mask_low_hit_num]
        truth_points = truth_points[mask_low_hit_num]
        truth_grouped = truth_grouped[mask_low_hit_num]
        num_after_low_hit_cut_ = len(features)
        num_after_low_hit_cut_hits = len(ak.flatten(features))

        # Record offsets
        offsets = ak.count(features, axis=1)[:, 0]
        offsets_cumsum = np.append(0, np.cumsum(offsets))
        features = np.array(ak.to_list(ak.flatten(features)))

        # Normalized truth and features
        if self.truth_mean is None:
            self.truth_mean = np.mean(truth_points, axis=0)+1E-10
            self.truth_std = np.std(truth_points, axis=0)+1E-10
        truth_points_normalized = (truth_points-self.truth_mean)/self.truth_std

        if self.hits_mean is None:
            self.hits_mean = np.mean(features, axis=0)+1E-10
            self.hits_std = np.std(features, axis=0)+1E-10
        hits_normalized = (features-self.hits_mean)/self.hits_std

        # Check that everything is correct
        assert len(offsets_cumsum) == len(truth_points)+1
        totlen = offsets_cumsum_with_empty_[-1]
        print(tabulate(
            [["Original root file:", len(offsets_cumsum_with_empty_), totlen, 100.0],
             ["Removed absorber hits:", len(offsets_without_absorber_cumsum), offsets_without_absorber_cumsum[-1], f"{offsets_without_absorber_cumsum[-1]/totlen*100:.2f}"],
             ["Summed over calo cells:", len(offsets_sum_calo), len(features_sum_calo), f"{len(features_sum_calo)/totlen*100:.2f}"],
             ["Removed low energy", len(offsets_sum_calo), num_after_low_E_cut, f"{num_after_low_E_cut/totlen*100:.2f}"],
             ["Removed small opening angle", num_after_angle_cut_, num_after_angle_cut_hits, f"{num_after_angle_cut_hits/totlen*100:.2f}"],
             ["Removed low hit count", num_after_low_hit_cut_, num_after_low_hit_cut_hits, f"{num_after_low_hit_cut_hits/totlen*100:.2f}"]],
            headers=["", "Number of ", "Number of hits", "Hits Ratio [%]"]))

        # Write to file
        if write:
            with uproot.recreate(f"{self.filename}_preprocessed.root") as file:
                file["Hits"] = {"hits_normalized": hits_normalized}
                file["Hits_row_splits"] = {"rowsplits": offsets_cumsum}
                file["Hits_offsets"] = {"offsets": offsets}
                file["Hits_parameters"] = {"hits_mean": self.hits_mean, "hits_std": self.hits_std}
                file["Truth"] = {"truth_normalized": truth_points_normalized}
                file["Truth_parameters"] = {"truth_mean": self.truth_mean, "truth_std": self.truth_std}
                file["Cuts"] = f"energy_cut,{energy_cut}\nlow_hits_cut,{low_hits_cut}\nopening_angle_cut,{opening_angle_cut}\n"

        if perfect_detector is True:
            # Perfect detector
            perfect_hits = []
            for x in np.concatenate([np.linspace(0, 250, 30), [400]]):
                perfect_hits.append(line(truth_grouped, 1, x))
                perfect_hits.append(line(truth_grouped, 2, x))
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
            if self.perfect_hits_mean is None:
                self.perfect_hits_mean = np.mean(perfect_hits, axis=0)+1E-10
                self.perfect_hits_std = np.std(perfect_hits, axis=0)+1E-10
            perfect_hits_normalized = (
                perfect_hits-self.perfect_hits_mean)/self.perfect_hits_std

            assert len(perfect_offsets_cumsum) == len(truth_points)+1

            # Write to file
            if write:
                with uproot.recreate(self.filename+"_idealized_preprocessed.root") as file:
                    file["Hits"] = {"hits_normalized": perfect_hits_normalized}
                    file["Hits_row_splits"] = {
                        "rowsplits": perfect_offsets_cumsum}
                    file["Hits_offsets"] = {"offsets": perfect_offsets}
                    file["Hits_parameters"] = {
                        "hits_mean": self.perfect_hits_mean,
                        "hits_std": self.perfect_hits_std}
                    file["Truth"] = {
                        "truth_normalized": truth_points_normalized}
                    file["Truth_parameters"] = {
                        "truth_mean": self.truth_mean, "truth_std": self.truth_std}

        return ak.unflatten(features, offsets), truth_points

    def _find_hits(self) -> ak.Array:
        keys = ["layerType", "cellID", "x", "y", "z", "E"]
        prefix = "/Output/fHits/fHits."
        raw_array = []

        print("[1/3] Extract from root file")
        for keys in keys:
            raw_array.append(self._extract_to_array(
                prefix+keys)[..., np.newaxis])
        raw_array = ak.concatenate(raw_array, axis=-1)
        return raw_array

    def _find_truth(self) -> ak.Array:
        """ Shape:
        ( x 12)
        """
        prefix = "/Output/fParticles/fParticles."
        keys = ["decayX", "decayY", "decayZ"]
        vertex = []

        for key in keys:
            vertex.append(self._extract_to_array(prefix+key)[..., np.newaxis])

        vertex = ak.concatenate(vertex, axis=-1)
        vertex = ak.flatten(vertex, axis=-1)
        vertex = ak.pad_none(vertex, target=3)
        vertex = ak.fill_none(vertex, 0.0)

        p1, p2 = self._find_momentum_photons()
        return ak.concatenate([p1, vertex, p2, vertex], axis=1)

    def _extract_to_array(self, key) -> ak.Array:
        """ This method extracts the  and hits for a given key in the root
        tree

        Returns shapes:
        (hits x [layerType, x, y, z, E])
        """
        with uproot.open(self.filename+".root") as data:
            return data[key].array(library="ak")

    def _find_offsets(self, df: pd.DataFrame) -> np.ndarray:
        return df.reset_index(level=1).index.value_counts().sort_index().to_numpy()

    def _find_momentum_photons(self):
        """ This function reads the momentum from the two founds directly from the hepmc file since the simulation
        currently does not records the daughter particles of the alp. The truth position however is correct in the
        simulation output file.
        Expects no header in the hepmc file
        """
        with open(f"/ceph/kschmidt/beamdump/alps/simulation/alp.hepmc3") as file:
            photons = file.readlines()

        p1_raw = [line.strip() for i, line in enumerate(photons) if (i+1) % 7 == 6]
        p2_raw = [line.strip() for i, line in enumerate(photons) if (i+1) % 7 == 0]

        p1, p2 = [], []
        for line in p1_raw:
            parts = line.split()
            p1.append([float(part) for part in [parts[6], parts[5], parts[4]]])
        for line in p2_raw:
            parts = line.split()
            p2.append([float(part) for part in [parts[6], parts[5], parts[4]]])

        return ak.Array(p1), ak.Array(p2)


if __name__ == "__main__":
    pi = PrepareInputs()

    features, truth = pi.perform("/ceph/kschmidt/beamdump/alps/simulation/alp.root", opening_angle_cut=0.65)

    

