import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import uproot
import sys
import os
from datetime import datetime
from typing import Type, TypeVar, Literal

today = datetime.today().strftime("%Y-%m-%d")


class Extract:
    """ Class which automatically extracts the data from the neural network
    output (.djctd file) and converts it to awkward arrays

    Notes
    -----
    Opens the file <model_dir><model_name>/Predicted/pred_Testing.djctd"

    Parameters
    ----------
    model_dir : str
        Directory where the models are stored
    model_name : str
        Specific name of the model output
    root_files_dir : str
        Directory of the root files used by the network

    Returns
    -------
    Data container of Type[TObject] with objects:
        self.predicted\n
        self.truth\n
        self.uncertainties (if provided, empty otherwise)\n
    And additional information to be passed to plotting tools
        self.model_dir\n
        self.model_name\n
    """

    def __init__(self,
                 model_dir="./nntr_models",
                 model_name="",
                 read_hits=False,
                 testing_root_files="./nntr_data/Raw/Testing.root"):

        self.model_dir = model_dir+"/"
        self.model_name = model_name
        self.nn_raw_data = {}
        self.predicted = self.nn_data(key="Predicted")
        self.truth = self.nn_data(key="Truth")
        self.testing_root_files = testing_root_files
        self.hits = []
        self.ATCCellNumCols = np.NaN
        self.ATCCellNumRows = np.NaN
        self.ATLayerNum = np.NaN
        self.CalorimeterThickness = np.NaN
        self.ATCellSideLength = np.NaN

        if "Uncertainties" in self.nn_raw_data.keys():
            self.uncertainties = self.nn_data(key="Uncertainties")
        else:
            self.uncertainties = []

        if read_hits:
            self.read_hits()

    def undo_normalization(self, array: np.array):
        """ Only temporary as this will be handled by conversion
        in DJC
        """
        print("Undo normalization, mean was", np.mean(array))
        with open("nntr_pd_data/4173685_tmp_Testing.pkl", "rb") as file:
            data = pickle.load(file)
            std = data["Std"]
            mean = data["Mean"]
            original_array = array*(std + 1E-10) + mean
            print("Mean is now", np.mean(original_array))
            return original_array

    def nn_data(self, key="Truth"):
        return self.nn_find_physical_variables(self.nn_read_data()[key])

    def nn_read_data(self):
        if self.model_name not in os.listdir(self.model_dir):
            print("No such model could be found at:")
            print(f"{self.model_dir}{self.model_name}")
        else:
            with open(f"{self.model_dir}/{self.model_name}/"
                      + "Predicted/pred_Testing.djctd", "rb") as file:
                self.nn_raw_data = pickle.load(file)
            return self.nn_raw_data

    def nn_find_physical_variables(self, array: np.array):

        # Temporary until new data un-normalized by DJC
        array = self.undo_normalization(array)
        # remove this block ^^^ afterwards

        if np.shape(array)[1] == 14:
            return {
                "p1": array[:, 0:3],
                "n1": array[:, 3],
                "v1": array[:, 4:7],
                "p2": array[:, 7:10],
                "n2": array[:, 10],
                "v2": array[:, 11:14]}
        elif np.shape(array)[1] == 12:
            return {
                "p1": array[:, 0:3],
                "v1": array[:, 3:6],
                "p2": array[:, 6:9],
                "v2": array[:, 9:12]}
        else:
            print("Wrong shape of array, is", np.shape(array))

    def read_hits(self):
        filename = self.testing_root_files

        with uproot.open(filename) as event:
            keys = ["layerType", "layerID", "cellID", "E", "x", "y", "z"]
            self.hits = {}
            for name in keys:
                self.hits[name] = event["Events"]["Output"]["fHits"][f"fHits.{name}"].array(library="ak")

            self.ATCCellNumCols = event["Events"]["ParametersNumericals"]["fATCCellNumCols"].array(library="np")[0]
            self.ATCCellNumRows = event["Events"]["ParametersNumericals"]["fATCCellNumRows"].array(library="np")[0]
            self.ATLayerNum = event["Events"]["ParametersNumericals"]["fNumberATLayers"].array(library="np")[0]
            self.CalorimeterThickness = event["Events"]["ParametersNumericals"]["fCalorimeterThickness"].array(library="np")[0]
            self.ATCellSideLength = event["Events"]["ParametersNumericals"]["fAbsorberSideLength"].array(library="np")[0]

        return None


TOutput = TypeVar("TOutput", bound=Extract)


class Plot:

    def savefig(savefigdir,
                data=Type[TOutput],
                plot_type="",
                obs=""):
        if not savefigdir:
            savefigdir = f"{data.model_dir}/{data.model_name}/Plots/"
        os.makedirs(savefigdir, exist_ok=True)
        plt.savefig(f"{savefigdir}/{plot_type}_{obs}.png", dpi=600)

    def histogram(data=Type[TOutput],
                  obs="v1",
                  xlabel="Distance from detector [mm]",
                  scalar_factor=1.0,
                  nbins=np.array([]),
                  show_uncertainties=True,
                  savefigdir="",
                  logscale=True) -> None:
        """ Plot a pair of histograms comparing the predicted and
        true distributions of a given quantity from the network
        output.

        Notes
        -----
        If uncertainties are included in 'data', they will be added
        automatically to the plots as horizontal errorbars.

        Parameters
        ----------

        data : Type[TOutput]
            Data extracted from the network using the Extract class
        obs : str
            Key in 'data' of the observable to be plotted, can be a
            single array (e.g. Energy) or a 3-vector (e.g. Momentum)
        xlabel : str
            Shared xlabel passed to matplotlib, should include units
        scalar_factor : float
            Factor for all quantities to account for unit conversions
            (e.g. MeV -> Gev)
        """

        predicted = data.predicted[obs]
        truth = data.truth[obs]
        uncertainties = data.uncertainties[obs]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
        fig.suptitle(f"Model {data.model_name}")

        if not nbins.any():
            xlim = (np.min([predicted, truth])*scalar_factor,
                    np.max([predicted, truth])*scalar_factor)
            nbins = np.linspace(xlim[0], xlim[1], 50)
        else:
            xlim = nbins[0], nbins[-1]
        ax1.set_xlim(xlim)
        ax2.set_xlim(xlim)

        label = obs
        if len(predicted[0]) == 3:
            label = [f"{obs}_{i}" for i in ["x", "y", "z"]]

        ax1.hist(predicted*scalar_factor,
                 bins=nbins,
                 histtype="step",
                 alpha=0.9,
                 label=label)

        def gaussian_uncertainties(ax):
            mean = np.mean(predicted, axis=0)
            errors = np.abs(np.mean(uncertainties, axis=0))
            height = []

            if np.atleast_2d(predicted).shape[0] != 1:
                prop_cycle = plt.rcParams['axes.prop_cycle']
                colors = prop_cycle.by_key()['color']

                for i in range(predicted.shape[1]):
                    h = np.histogram(predicted[..., i], bins=nbins)
                    height = h[0][h[0].argmax()]
                    ax.scatter(x=mean[i]*scalar_factor,
                               y=height,
                               color=colors[i])
                    ax.errorbar(x=mean[i]*scalar_factor,
                                y=height,
                                xerr=errors[i]*scalar_factor,
                                label=f"$\\Delta {label[i]}$",
                                ls="",
                                color=colors[i])

            else:
                h = np.histogram(predicted, bins=nbins)
                height = h[0][h[0].argmax()]
                ax.scatter(x=mean*scalar_factor,
                           y=height)
                ax.errorbar(x=mean*scalar_factor,
                            y=height,
                            xerr=errors*scalar_factor,
                            label="Predicted Error",
                            ls="")

        if uncertainties.any() and show_uncertainties:
            gaussian_uncertainties(ax1)

        ax1.set_title('Predicted')

        ax2.hist(truth*scalar_factor,
                 bins=nbins,
                 histtype="step",
                 alpha=0.9,
                 label=label)
        ax2.set_title('Truth')

        for ax in (ax1, ax2):
            ax.set_xlabel(xlabel)
            ax.set_ylabel('Number of occurences')
            if logscale:
                ax.set_yscale('log')
            ax.legend(loc="upper right")

        plt.tight_layout()
        Plot.savefig(savefigdir, data, "hist", obs)
        return None

    def scatter(data=Type[TOutput],
                obs="v1",
                axis=("x", "y"),
                units="mm",
                xlim=(),
                ylim=(),
                scalar_factor=1,
                savefigdir="") -> None:

        """ Produce a scatter plot in the plane defined by 'axis' to
        compare the predicted and true distributions of 'obs'.

        Parameters
        ----------
        data : Type[TOutput]
            Data produced by the Extract class
        obs : str
            Name of the observable ('v1' or 'v2) to be plotted
        units : str
        scalar_factor : float
            Factor for all quantities to account for unit conversions
            (e.g. MeV -> Gev)
        """

        p = data.predicted[obs]
        t = data.truth[obs]
        ax = {"x": 0, "y": 1, "z": 2}

        plt.figure()
        plt.scatter(x=p[..., ax[axis[0]]] * scalar_factor,
                    y=p[..., ax[axis[1]]] * scalar_factor,
                    alpha=0.1, marker=".",
                    label="Predicted")
        plt.scatter(x=t[..., ax[axis[0]]] * scalar_factor,
                    y=t[..., ax[axis[1]]] * scalar_factor,
                    alpha=0.3, marker=".",
                    label="True")
        plt.legend()
        if xlim:
            plt.xlim(xlim)
        if ylim:
            plt.ylim(ylim)
        plt.xlabel(f"{axis[0]} [{units}]")
        plt.ylabel(f"{axis[1]} [{units}]")
        plt.title(f"Model {data.model_name}")
        Plot.savefig(savefigdir, data, f"scatter_{axis[0]}_{axis[1]}", obs)
        return None

    def df(data=Type[TOutput],
           mean=False,
           eventID=0):

        predicted = data.predicted
        truth = data.truth
        uncertainties = data.uncertainties
        iterables = [["Mean", "Uncertainties"],
                     ["Predicted", "Truth"]]
        obs = predicted.keys()
        if mean:
            header = pd.MultiIndex.from_product(iterables=iterables,
                                                names=["DataType", "Property"])
            df = pd.DataFrame(index=obs, columns=header)
            for i, ob in enumerate(obs):
                df.loc[ob, ("Mean", "Predicted")] = np.mean(predicted[ob],
                                                            axis=0)
                df.loc[ob, ("Uncertainties", "Predicted")] = np.mean(
                                                        uncertainties[ob],
                                                        axis=0)
                df.loc[ob, ("Mean", "Truth")] = np.mean(truth[ob], axis=0)
                df.loc[ob, ("Uncertainties", "Truth")] = np.std(truth[ob],
                                                                axis=0)
        else:
            header = pd.MultiIndex.from_product(iterables=[[eventID],
                                                           ["Predicted",
                                                            "Truth"]],
                                                names=["EventID", "Property"])
            df = pd.DataFrame(index=obs, columns=header)
            for i, ob in enumerate(obs):
                df.loc[ob, (eventID, "Predicted")] = predicted[eventID, i]
                df.loc[ob, (eventID, "Truth")] = truth[eventID, i]
        return df

    def tracks(data=Type[TOutput],
               eventID=0,
               axis="x",
               savefigdir=""):
        """ Method which calls the FullTrackReco class and plots the hits as
        well as the true and predicted trajectory of the photons based on the
        output from the network
        """
        if not data.hits:
            print("Error: Root files have not been extracted " +
                  "yet. Use Extract.read_hits() or set " +
                  "'read_hits' to True when using Extract().")

        ftr = FullTrackReco(data, eventID)
        ftr.plot_mpl(axis=axis)
        Plot.savefig(savefigdir, data, "tracks", eventID)
        return None


class FullTrackReco:
    def __init__(self,
                 data=Type[TOutput],
                 eventID=0,
                 particle="photon",
                 zlim=-10):
        self.nn_raw_data = []
        self.particle = particle
        self.zlim = zlim
        self.eventID = eventID
        self.data = data
        self.ATLayerNum = data.ATLayerNum
        self.ATCCellNumCols = data.ATCCellNumCols
        self.ATCCellNumRows = data.ATCCellNumRows
        self.ATCellSideLength = data.ATCellSideLength
        self.CalorimeterThickness = data.CalorimeterThickness

        if self.ATLayerNum <= 5:
            self.colors = ["purple", "blue", "darkred", "red", "darkorange"]
        else:
            self.colors = ["lightblue"]*self.ATLayerNum

        if self.data.hits:
            hits_event = {}
            for name in self.data.hits.keys():
                hits_event[name] = self.data.hits[name][eventID]
            self.df = pd.DataFrame(hits_event)

    def tracker_hits(self, layerID):
        """ Find the tracker hits for each layer
        """
        tracker = self.df[self.df["layerType"] == 1]
        tracker_layer = tracker[tracker["layerID"] == layerID]
        return tracker_layer

    def plot_calo(self):
        """ Heatmap of the Calorimeter energy deposition with tracker hits"""
        def photon_energy(momentum):
            assert len(momentum) == 3
            return np.linalg.norm(momentum)

        self.fig = plt.figure(figsize=(6, 5))
        self.ax = self.fig.subplots()

        # Calo hits
        calo = self.df[self.df["layerType"] == 2]
        cg = np.zeros(self.ATCCellNumRows*self.ATCCellNumCols)
        cg[calo["cellID"]] = 1e3*calo["E"]  # Are in GeV in the root file
        calo_colormesh = np.reshape(cg, (self.ATCCellNumRows,
                                         self.ATCCellNumCols))
        width = self.ATCellSideLength*np.arange(
                                        -np.floor(self.ATCCellNumCols/2),
                                        np.ceil(self.ATCCellNumCols/2))
        height = self.ATCellSideLength*np.arange(
                                        -np.floor(self.ATCCellNumRows/2),
                                        np.ceil(self.ATCCellNumRows/2))
        im = self.ax.pcolormesh(width,
                                height,
                                calo_colormesh,
                                cmap="GnBu",
                                alpha=0.7,
                                edgecolors="lightblue",
                                linewidth=0.5)
        plt.colorbar(mappable=im,
                     pad=0.01,
                     label="Calorimeter Energy Deposition [MeV]")

        # Tracker hits
        for layerID in range(0, self.ATLayerNum):
            trh = self.tracker_hits(layerID)

            if self.ATLayerNum <= 5:
                label = f"Tracking Layer {layerID}"
            else:
                label = None

            self.ax.scatter(x=trh["x"].values,
                            y=trh["y"].values,
                            marker=".",
                            c=self.colors[layerID],
                            label=label)

        self.energy = 1e3*np.sum(calo["E"])
        t = self.ax.text(x=0.50,
                         y=0.88,
                         s=f"{self.particle}, {self.ATLayerNum} AT layers",
                         transform=self.ax.transAxes)
        t.set_bbox(dict(facecolor="white", alpha=0.5, edgecolor="k"))

        self.ax.set_xlim(width[0]-20, width[-1]+20)
        self.ax.set_ylim(height[0]-20, height[-1]+20)
        self.ax.set_xlabel("x [mm]")
        self.ax.set_ylabel("y [mm]")
        plt.legend(loc="upper left")
        plt.title("Detector Hits")
        plt.tick_params(direction="in")

    def plot_mpl(self,
                 axis: Literal["x", "y"] = "x",
                 scalar_factor=1,
                 xlim=(),
                 ylim=()):

        fig, plot = plt.subplots(figsize=(6, 6))
        ax = {"x": 0, "y": 1}[axis]

        # Trajectories
        p = self.data.predicted
        t = self.data.truth
        units = scalar_factor
        pred1 = np.stack((p["p1"][self.eventID]*units,
                          p["v1"][self.eventID]*units), axis=-1)
        pred2 = np.stack((p["p2"][self.eventID]*units,
                          p["v2"][self.eventID]*units), axis=-1)
        true1 = np.stack((t["p1"][self.eventID]*units,
                          t["v1"][self.eventID]*units), axis=-1)
        true2 = np.stack((t["p2"][self.eventID]*units,
                          t["v2"][self.eventID]*units), axis=-1)
        print(pred1)

        # Predicted
        plot.plot(pred1[2], pred1[ax],
                  label='Reconstructed photon 1',
                  color="lightgreen")
        plot.plot(pred2[2], pred2[ax],
                  label='Reconstructed photon 2',
                  color="green")

        # Truth
        plot.plot(true1[2], true1[ax],
                  color="black",
                  alpha=0.8)
        plot.plot(true2[2], true2[ax],
                  label="True photons",
                  color="black",
                  alpha=0.8)

        # Tracker
        for layerID in range(0, self.ATLayerNum):
            trh = self.tracker_hits(layerID)

            if self.ATLayerNum <= 5:
                label = f"Tracking Layer {layerID}"
            else:
                label = None

            if not trh["z"].empty:
                plot.scatter(x=trh["z"].values,
                             y=trh[axis].values,
                             marker=".",
                             c=self.colors[layerID],
                             label=label)
                plot.vlines(x=trh["z"].values,
                            ymin=-self.ATCellSideLength*self.ATCCellNumRows/2,
                            ymax=self.ATCellSideLength*self.ATCCellNumCols/2,
                            color=self.colors[layerID],
                            alpha=0.2)
            else:
                pass

        # Calorimeter
        self.calo = self.df[self.df["layerType"] == 2]
        if not self.calo["z"].empty:
            plot.vlines(x=self.calo["z"],
                        ymin=self.calo[axis]-self.ATCellSideLength,
                        ymax=self.calo[axis]+self.ATCellSideLength,
                        alpha=self.calo["E"]/np.max(self.calo["E"])*0.7,
                        colors="k")
            self.calo_center = self.calo["z"].iloc[0]
        else:
            self.calo_center = 450
            pass

        detector = patches.Rectangle(
                          xy=(0, -self.ATCellSideLength*self.ATCCellNumRows/2),
                          width=+self.CalorimeterThickness/2,
                          height=self.ATCellSideLength*self.ATCCellNumRows,
                          color="grey",
                          alpha=0.1)
        plot.add_patch(detector)

        # Plotting parameters
        if not xlim:
            plot.set_xlim(-10 + t["v1"][self.eventID][2]*1E3,
                          self.calo_center + self.CalorimeterThickness/2+10)
        else:
            plot.set_xlim(xlim)

        if not ylim:
            plot.set_ylim(-self.ATCellSideLength*self.ATCCellNumCols,
                          self.ATCellSideLength*self.ATCCellNumRows)
        else:
            plot.set_ylim(ylim)

        plot.set_ylabel(axis + " [mm]")
        plot.set_xlabel("z [mm]")
        plot.legend(loc=(1.05, 0.55))
        return None


if __name__ == "__main__":
    model_name = sys.argv[1]

    read_hits = False
    data = Extract(model_dir="./nntr_models",
                   model_name=model_name,
                   read_hits=read_hits)

    Plot.histogram(data=data,
                   obs="p1",
                   xlabel="Normalized Momentum",
                   scalar_factor=1,
                   nbins=np.linspace(-5, 5, 100))

    Plot.histogram(data=data,
                   obs="v1",
                   xlabel="Vertex Position [mm]",
                   scalar_factor=1E3,
                   nbins=np.linspace(-2000, 100, 100))

    Plot.scatter(data=data,
                 obs="v1",
                 xlim=(-1, 1),
                 ylim=(-1, 1),
                 scalar_factor=1,
                 units="m")

    Plot.scatter(data,
                 "v1",
                 xlim=(-2, 0),
                 ylim=(-2, 2),
                 scalar_factor=1,
                 axis=("z", "x"),
                 units="m")

    #Plot.tracks(data=data, eventID=0)
