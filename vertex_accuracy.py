import pandas as pd
import numpy as np
import sys, os
import matplotlib.pyplot as plt
import uncertainties as unc
from plotting import Results
plt.style.use("belle2")


def plot_add_info(ax: plt.Axes, text="Demonstration testing set"):
    ax.text(
        0.02, 0.94,
        "Beamdump",
        transform=ax.transAxes,
        fontsize=16,
        fontweight="bold")
    ax.text(
        0.02, 0.8,
        "Photon Reconstruction 2024\n"
        + f"{text}",
        transform=ax.transAxes,
        fontsize=12)


if __name__ == "__main__":
    ta_distance = sys.argv[1]
    region = ""

    check_point_file = f"/work/kschmidt/DeepJetCore/TrackReco_DeepJetCore/HGCalML/nntr_models/normal_detector/ta_distance_sweep/tad_{ta_distance}/Predicted/VertexAccuracy{region}.csv"
    if os.path.exists(check_point_file):
        data = pd.read_csv(check_point_file).to_numpy()[:, 1]
    else:
        results = Results()
        hits, truth, predicted = results.hits, results.truth, results.predicted
        df = results.create_df()
        data = df["VertexAccuracy"]
        data.to_csv(check_point_file)

    # Plotting
    limits = (-800, 800)
    bins = np.linspace(limits[0], limits[1], 64+1)
    x = np.linspace(limits[0], limits[1], 2000)

    def FWHM(x: np.array, y: np.array):
        def find_nearest(array, value):
            array = np.asarray(array)
            index = (np.abs(array - value)).argmin()
            return index

        max = find_nearest(y, np.max(y))
        fwhmheight = np.max(y) / 2
        x1 = x[find_nearest(y[:max], fwhmheight)]
        x2 = x[max + find_nearest(y[max:], fwhmheight)]
        fwhm = x2 - x1
        return fwhm, fwhmheight, x1, x2

    fig, ax = plt.subplots()
    plot_add_info(ax, f"TA distance : {ta_distance} mm")
    # Histogram
    hist_height, hist_bins = np.histogram(data, bins)
    bin_centers = 0.5*(hist_bins[1:] + hist_bins[:-1])
    plt.plot(
        bin_centers,
        hist_height,
        marker=None,
        drawstyle="steps-mid",
        label="MC Data",
        color="black",
    )
    ax.bar(
        bin_centers,
        height=2*hist_height**0.5,
        width=hist_bins[1]-hist_bins[0],
        bottom=hist_height - hist_height**0.5,
        color="black",
        hatch="///////",
        fill=False,
        lw=0,
        label="Stat. unc.",
            )
    # Density
    counts, bins_fit = np.histogram(data, x)
    density = np.sum(counts)*np.diff(bins_fit)[0]/len(bins)*len(x)
    # FWHM
    fwhm, fwhm_height, x1, x2 = FWHM(x, y*density)
    plt.hlines(fwhm_height, x1, x2, color="red", label=f"FWHM={fwhm:.2f} mm")
    # Fit
    label = "DSCB Fit Parameters:\n"
    for key in parameter.keys():
        label = label+f" {key}={parameter[key]:.2f}\n"
    label = label.rstrip("\n")
    plt.plot(
        x,
        y*density,
        label=label
    )
    print("Goodness of fit:", gof)
    plt.xlim(bins[0], bins[-1])
    plt.ylim(0, )
    plt.xlabel(r"$V_{1,z}^{true} - V_{1,z}^{pred}$"+"[mm]")
    plt.ylabel(f"Counts / ({(bins[1]-bins[0]):.1f} [mm])")

    plt.legend(loc="upper right", fontsize=10)
    plt.savefig(f"/work/kschmidt/DeepJetCore/TrackReco_DeepJetCore/HGCalML/nntr_models/normal_detector/ta_distance_sweep/tad_{ta_distance}/Plots/vertex_accuracy")
    plt.show()
