import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import uproot
from icecream import ic
import os
import awkward as ak
import logging
from datetime import datetime
from typing import Type, TypeVar, Literal
logger = logging.getLogger(__name__)
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

    def __init__(
            self,
            model_dir="./nntr_models",
            model_name="",
            testing_root_files="./nntr_data/Raw/Testing.root",
            predicted_file="Predicted/pred_Testing.djctd",
            read_uncertainties=True):

        self.model_dir = model_dir+"/"
        self.predicted_file = predicted_file
        self.model_name = model_name
        self.testing_root_files = testing_root_files
        self.hits = []
        self.ATCCellNumCols = 5
        self.ATCCellNumRows = 5
        self.ATLayerNum = 5
        self.CalorimeterThickness = 250
        self.ATCellSideLength = 40
        self.read_uncertainties = read_uncertainties
        self.savefigdir = ""

        # Read hits
        with uproot.open(self.testing_root_files) as file:
            self.hits_normalized = file["Hits"]["hits_normalized"].arrays()["hits_normalized"]
            self.offsets_cumsum = file["Hits_row_splits"].arrays()["rowsplits"]
            self.offsets = file["Hits_offsets"].arrays()["offsets"]
            self.hits_mean = file["Hits_parameters"]["hits_mean"].arrays()["hits_mean"]
            self.hits_std = file["Hits_parameters"]["hits_std"].arrays()["hits_std"]
            self.truth_normalized = file["Truth"]["truth_normalized"].arrays()["truth_normalized"]
            self.truth_mean = file["Truth_parameters"]["truth_mean"].arrays()["truth_mean"]
            self.truth_std = file["Truth_parameters"]["truth_std"].arrays()["truth_std"]

        self.hits_flattened = self.hits_normalized*self.hits_std+self.hits_mean
        self.hits = ak.unflatten(self.hits_flattened, self.offsets)
        self.truth_array = self.truth_normalized*self.truth_std+self.truth_mean
        self.truth = {}
        self.truth["A1"] = self.truth_array[:, 0:3]
        self.truth["B1"] = self.truth_array[:, 3:6]
        self.truth["A2"] = self.truth_array[:, 6:9]
        self.truth["B2"] = self.truth_array[:, 9:12]
        self.truth["V1"] = self.truth_array[:, 12:15]
        self.truth["V2"] = self.truth_array[:, 15:18]

        # Read prediction and truth
        if self.model_name not in os.listdir(self.model_dir):
            logger.error(
                "No such model could be found at:\n" +
                f"{self.model_dir}/{self.model_name}\n" +
                "Available models are:", os.listdir(self.model_dir))
        else:
            with open(f"{self.model_dir}/{self.model_name}/{self.predicted_file}", "rb") as file:
                prediction_all = pickle.load(file)["Predicted"]
                self.predicted_raw = prediction_all[0]
                predicted = self.predicted_raw[:, 0:18]*self.truth_std + self.truth_mean

                if self.read_uncertainties and len(prediction_all[0][0]) == 36:
                    self.uncertainties = prediction_all[0][:, 18:36]
                    print("Caution: when training ln_sigma, rescale with np.exp(uncertainties). Also take care of undoing any normalization")

            self.predicted = {}
            self.predicted["A1"] = predicted[:, 0:3]
            self.predicted["B1"] = predicted[:, 3:6]
            self.predicted["A2"] = predicted[:, 6:9]
            self.predicted["B2"] = predicted[:, 9:12]
            self.predicted["V1"] = predicted[:, 12:15]
            self.predicted["V2"] = predicted[:, 15:18]


TOutput = TypeVar("TOutput", bound=Extract)


class Plot:

    def savefig(
            savefigdir,
            data=Type[TOutput],
            plot_type="",
            obs=""):
        if not savefigdir:
            if data.savefigdir:
                savefigdir = data.savefigdir
            else:
                savefigdir = f"{data.model_dir}/{data.model_name}/Plots/"
        os.makedirs(savefigdir, exist_ok=True)
        plt.savefig(f"{savefigdir}/{plot_type}_{obs}.png", dpi=600)

    def scatter(
            data=Type[TOutput],
            obs="v1",
            axis=("x", "y"),
            scalar_factor=1) -> None:
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
        fig, plot = plt.subplots(figsize=(6, 6))
        p = data.predicted[obs]
        t = data.truth[obs]
        ax = {"x": 0, "y": 1, "z": 2}

        plot.scatter(
            x=p[..., ax[axis[0]]] * scalar_factor,
            y=p[..., ax[axis[1]]] * scalar_factor,
            alpha=0.1,
            marker=".",
            label="Predicted")
        plot.scatter(
            x=t[..., ax[axis[0]]] * scalar_factor,
            y=t[..., ax[axis[1]]] * scalar_factor,
            alpha=0.3,
            marker=".",
            label="True")
        return plot

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
            header = pd.MultiIndex.from_product(
                iterables=iterables,
                names=["DataType", "Property"])
            df = pd.DataFrame(columns=header)
            
            for i, ob in enumerate(obs):
                for a, axis in enumerate({"x": 0, "y": 1, "z": 2}):
                    df.loc[ob+axis, ("Mean", "Predicted")] = np.mean(predicted[ob], axis=0)[a]
                    df.loc[ob+axis, ("Uncertainties", "Predicted")] = np.mean(uncertainties[ob], axis=0)[a]
                    df.loc[ob+axis, ("Mean", "Truth")] = np.mean(truth[ob], axis=0)[a]
                    df.loc[ob+axis, ("Uncertainties", "Truth")] = np.std(truth[ob], axis=0)[a]
        else:
            header = pd.MultiIndex.from_product(
                iterables=[[eventID], ["Predicted", "Truth"]],
                names=["EventID", "Property"])
            df = pd.DataFrame(index=obs, columns=header)

            for i, ob in enumerate(obs):
                df.loc[ob, (eventID, "Predicted")] = predicted[eventID, i]
                df.loc[ob, (eventID, "Truth")] = truth[eventID, i]

        return df


class FullTrackReco:
    def __init__(
            self,
            data=Type[TOutput],
            eventID=0,
            particle="photon",
            zlim=-10):
        self.particle = particle
        self.predicted = {}
        self.truth = {}
        self.zlim = zlim
        self.eventID = eventID
        self.data = data
        self.ATLayerNum = data.ATLayerNum
        self.ATCCellNumCols = data.ATCCellNumCols
        self.ATCCellNumRows = data.ATCCellNumRows
        self.ATCellSideLength = data.ATCellSideLength
        self.CalorimeterThickness = data.CalorimeterThickness

        if len(self.data.hits) != 0:
            self.df = pd.DataFrame(
                self.data.hits[eventID],
                columns=["layerType", "x", "y", "z", "E"])

        for key in self.data.predicted.keys():
            self.predicted[key] = self.data.predicted[key][eventID]
            self.truth[key] = self.data.truth[key][eventID]

    def plot_calo(self):
        """ Heatmap of the Calorimeter energy deposition with tracker hits"""
        fig, ax = plt.subplots(figsize=(6, 6))

        # Calo hits
        calo = self.df[self.df["layerType"] == 2]
        ic(calo["E"].to_numpy())
        calo_colormesh = np.zeros((self.ATCCellNumCols, self.ATCCellNumRows))
        ic(calo_colormesh)
        coord = calo[["x", "y"]].to_numpy().astype(int)

        width = self.ATCellSideLength*np.arange(
            -np.floor(self.ATCCellNumCols/2),
            +np.ceil(self.ATCCellNumCols/2))
        height = self.ATCellSideLength*np.arange(
            -np.floor(self.ATCCellNumRows/2),
            +np.ceil(self.ATCCellNumRows/2))
        im = ax.pcolormesh(
            width,
            height,
            calo_colormesh,
            cmap="GnBu",
            alpha=0.7,
            edgecolors="lightblue",
            linewidth=0.5)
        plt.colorbar(
            mappable=im,
            pad=0.01,
            label="Calorimeter Energy Deposition [MeV]")

        # Tracker hits

        ax.scatter(
            x=self.df[self.df["layerType" == 1]]["x"].values,
            y=self.df[self.df["layerType" == 1]]["y"].values,
            marker=".",
            label="Tracker")

        self.energy = 1e3*np.sum(calo["E"])
        t = ax.text(
            x=0.50,
            y=0.88,
            s=f"{self.particle}, {self.ATLayerNum} AT layers",
            transform=ax.transAxes)
        t.set_bbox(dict(facecolor="white", alpha=0.5, edgecolor="k"))

        ax.set_xlim(width[0]-20, width[-1]+20)
        ax.set_ylim(height[0]-20, height[-1]+20)
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        plt.legend(loc="upper left")
        plt.title("Detector Hits")
        plt.tick_params(direction="in")
        return fig, ax

    def plot_mpl(
            self,
            axis: Literal["x", "y"] = "x",
            show_absorber=False,
            scale_hits_with_energy=False):

        fig, plot = plt.subplots(figsize=(6, 6))
        ax = {"x": 0, "y": 1}[axis]

        p = self.predicted
        t = self.truth
        
        # Predicted
        plot.scatter(
            x=[p["V1"][2], p["A1"][2], p["B1"][2]],
            y=[p["V1"][ax], p["A1"][ax], p["B1"][ax]],
            color="green"
        )
        plot.scatter(
            x=[p["V2"][2], p["A2"][2], p["B2"][2]],
            y=[p["V2"][ax], p["A2"][ax], p["B2"][ax]],
            color="lightgreen"
        )
        plot.plot(
            [p["V1"][2], p["B1"][2]],
            [p["V1"][ax], p["B1"][ax]],
            color="green",
            linestyle="dashed",
            alpha=0.5,
            label="Predicted photon 1"
        )
        plot.plot(
            [p["V2"][2], p["B2"][2]],
            [p["V2"][ax], p["B2"][ax]],
            color="lightgreen",
            linestyle="dashed",
            alpha=0.5,
            label="Predicted photon 2"
        )
        # True
        plot.scatter(
            x=[t["V1"][2], t["A1"][2], t["B1"][2]],
            y=[t["V1"][ax], t["A1"][ax], t["B1"][ax]],
            color="k",
        )
        plot.scatter(
            x=[t["V2"][2], t["A2"][2], t["B2"][2]],
            y=[t["V2"][ax], t["A2"][ax], t["B2"][ax]],
            color="k",
        )
        plot.plot(
            [t["V1"][2], t["A1"][2], t["B1"][2]],
            [t["V1"][ax], t["A1"][ax], t["B1"][ax]],
            color="k",
            linestyle="dashed",
            alpha=0.5,
            label="True photon 1"
        )
        plot.plot(
            [t["V2"][2], t["A2"][2], t["B2"][2]],
            [t["V2"][ax], t["A2"][ax], t["B2"][ax]],
            color="k",
            linestyle="dashed",
            alpha=0.5,
            label="True photon 2"
        )

        
        # Absorber
        if show_absorber:
            absorber = self.df[self.df["layerType"] == 0]

            if scale_hits_with_energy:
                absorber_hits_size = absorber["E"]/np.max(absorber["E"])*1E3
            else:
                absorber_hits_size = None

            if not absorber["z"].empty:
                plot.scatter(
                    x=absorber["z"].values,
                    y=absorber[axis].values,
                    marker=".",
                    color="k",
                    label="Absorber",
                    s=absorber_hits_size)

        # Tracker
        tracker = self.df[self.df["layerType"] == 1]
        plot.scatter(
            x=tracker["z"].values,
            y=tracker[axis].values,
            marker=".",
            label="Tracker")
        plot.vlines(
            x=tracker["z"].values,
            ymin=-self.ATCellSideLength*self.ATCCellNumRows/2,
            ymax=self.ATCellSideLength*self.ATCCellNumCols/2,
            alpha=0.1)


        # Calorimeter
        self.calo = self.df[self.df["layerType"] == 2]
        if not self.calo["z"].empty:
            plot.vlines(
                x=self.calo["z"],
                ymin=self.calo[axis]-self.ATCellSideLength,
                ymax=self.calo[axis]+self.ATCellSideLength,
                alpha=0.7,
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

        return fig, plot


# Plotting tools

class PlottingWrapper:

    def plot_add_info(ax: plt.Axes, detector_type="Idealized detector"):
        ax.text(
            0.02, 0.94,
            "Beamdump",
            transform=ax.transAxes,
            fontsize=16,
            fontweight="bold")
        ax.text(
            0.02, 0.8,
            "Track Reconstruction 2023\n"
            + f"({detector_type})\n"
            + r"$10^6$"+" events, "+r"$E_0 = 1-2$"+" GeV",
            transform=ax.transAxes,
            fontsize=12)
        

    def plot_hits_per_events_histogram(data):
        if data.model_name == "idealized_detector":
            print("Skipped plotting hits per events histograms as the idealized detector version has constant"+
                "hits per events")
            pass
        elif data.model_name == "normal_detector":
            fig, ax = plt.subplots(1, 1, figsize=(8, 7))
            x = data.hits["x"][data.hits["layerType"] == 0]
            l = []
            for i in range(len(x)):
                l.append(len(x[i]))

            ax.hist(l, bins=100)
            ax.set_title("Normal detector - Hits per events")
            ax.vlines(np.mean(l), 0, 400, color="darkred", label=f"Mean = {round(np.mean(l))}")
            ax.set_xlabel("hits per event")
            ax.set_ylabel("Counts / 100 events")
            ax.legend("upper right")
            plt.savefig(data.savefigdir+"events_histogram")
            return ax

    def plot_point(data, obs="A1"):
        xlim = (-500, 500)
        bins = np.linspace(xlim[0], xlim[1], 100)
        fig, ax = plt.subplots(1, 1, figsize=(8, 7))
        density = True

        ax.hist(
            (data.predicted[obs][:, 2]*1E3 - data.truth[obs][:, 2]*1E3),
            bins=bins,
            histtype="step",
            label=r"$\Delta V_{1, z}$",
            density=density)

        PlottingWrapper.plot_add_info(ax)
        ax.set_xlabel(r"$V_1^{pred} - V_1^{true}$"+" [mm]", loc="right")
        ax.set_ylabel(f"Events / ({np.round(bins[1]-bins[0], 0)} mm)", loc="top")
        ax.legend(loc="upper right")
        ax.set_xlim(xlim)
        ax.set_yscale("linear")
        plt.savefig(data.savefigdir+"residuals_v1")
        return ax


    def plot_v1_z(data):
        fig, ax = plt.subplots(1, 1, figsize=(8, 7))
        xlim = (-1.5E3, 0.1E3)
        bins = np.linspace(xlim[0], xlim[1], 50)
        ax.hist(data.predicted["v1"][:, 2]*1E3, bins=bins, histtype="step", label="Predicted")
        ax.hist(data.truth["v1"][:, 2]*1E3, bins=bins, histtype="step", label="Truth")
        ax.set_xlabel(r"$V_z$ [mm]", loc="right")
        ax.set_ylabel(f"Events / ({np.round(bins[1]-bins[0], 0)} mm)", loc="top")
        ax.set_title("Vertex Position along detector axis")
        PlottingWrapper.plot_add_info(ax)
        ax.legend(loc="upper right")
        ax.set_ylim(0, 1600)
        plt.savefig(data.savefigdir+"v1z")
        return ax


    def plot_v1_z_pred_vs_true(data):
        t = data.truth["v1"][:, 2]*1E3
        p = data.predicted["v1"][:, 2]*1E3

        fig, ax = plt.subplots(1, 1, figsize=(8, 7))

        h = ax.hist2d(t, p, bins=50, cmap="OrRd")
        PlottingWrapper.plot_add_info(ax)
        ax.set_xlim(-1100, -100)
        ax.set_ylim(-1100, -100)
        ax.set_xlabel(r"$V_{1, z}^{true}$ [mm]", loc="right")
        ax.set_ylabel(r"$V_{1, z}^{pred}$ [mm]", loc="top")
        fig.colorbar(h[3], ax=ax, label="Counts")
        ax.axline(
            (-1000, -1000),
            (0, 0),
            color="black",
            linestyle="--",
            label=r"$y=x$")
        plt.savefig(data.savefigdir+"v1_z_pred_vs_true_hist2d")
        return ax


    def plot_tracks(data, eventID, axis="x"):
        """ Method which calls the FullTrackReco class and plots the hits as
        well as the true and predicted trajectory of the photons based on the
        output from the network
        """
        if len(data.hits) == 0:
            logger.error(
                "Error: Root files have not been extracted yet. Use Extract.read_hits() or set " +
                "'read_hits' to True when using Extract().")

        ftr = FullTrackReco(data, eventID)

        fig, plot = ftr.plot_mpl(
            scale_hits_with_energy=False,
            show_absorber=True,
            axis=axis
        )
        plot.set_xlabel("z [mm]", loc="right")
        plot.set_ylabel(f"{axis} [mm]", loc="top")
        PlottingWrapper.plot_add_info(plot)
        #plot.legend(loc="lower left")
        plot.set_xlim(-1500, 550)
        plot.set_ylim(-140, 150)
        plot.set_title("Detector sideview with two photon vertex")
        plot.legend(loc="center left")
        fig.set_size_inches(8, 6)
        plt.savefig(f"{data.savefigdir}/Tracks/event_{eventID}_{axis}")
        return plot
    
    def plot_calo(data, eventID):
        ftr = FullTrackReco(data, eventID)
        ftr.plot_calo()
        plt.savefig(f"{data.savefigdir}/Calo/event_{eventID}")
        return ftr


if __name__ == "__main__":
    
    detector_type = "normal_detector"
    model_name = "11_30"
    
    
    if detector_type == "idealized_detector":
        data = Extract(
            model_dir=f"./nntr_models/idealized_detector/garnet/",
            model_name=model_name,
            testing_root_files="./nntr_data/idealized_detector/Raw/Testing.root",
            read_uncertainties=True)

        with uproot.open("./nntr_data/idealized_detector/Raw/Testing.root") as file:
            hits = {}
            for key in ['layerType', 'x', 'y', 'z', 'E']:
                hits[key] = file["Events"]["Output"]["fHits"][f"fHits.{key}"].array(library="ak")
            hits["cellID"] = np.full(np.shape(hits["x"]), 0)
            hits["layerID"] = np.repeat(np.repeat([np.arange(0, 51)], 2, axis=1), len(hits["x"]), axis=0)

        data.hits = hits
        data.ATLayerNum = 50
        data.ATCCellNumCols = 5
        data.ATCCellNumRows = 5
        data.ATCellSideLength = 40
        data.CalorimeterThickness = 10

    elif detector_type == "normal_detector":
        data = Extract(
            model_dir=f"./nntr_models/{detector_type}/garnet",
            model_name=model_name,
            testing_root_files="./nntr_data/normal_detector/Raw/Testing_preprocessed.root",
            predicted_file="Predicted/pred_Testing_preprocessed.djctd",
            read_uncertainties=True)

    data.savefigdir = data.model_dir+data.model_name+"/Plots/"
    os.makedirs(data.savefigdir+"/Tracks/", exist_ok=True)

    PlottingWrapper.plot_hits_per_events_histogram(data)
    PlottingWrapper.plot_point(data, obs="A1")
    PlottingWrapper.plot_point(data, obs="B1")
    #PlottingWrapper.plot_v1_z(data)
    #PlottingWrapper.plot_v1_z_pred_vs_true(data)
    eventID = np.random.default_rng().choice(len(data.truth["A1"]))
    PlottingWrapper.plot_tracks(data, eventID, "x")
    PlottingWrapper.plot_tracks(data, eventID, "y")
    PlottingWrapper.plot_calo(data, eventID)

