from zfitwrapper import utilities, Models, Fitter
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import uncertainties as unc
plt.style.use("belle2")


class FitDSCB():
    def dofit(array, limits=(-800, 800)):
        params_dscb = {"Model": {
            "pdf": "DoubleCB", "parameter": {
                "mu": {"name": "mu", "value": 0, "lower": -3, "upper": 0, "floating": True},
                "sigma": {"name": "sigma", "value": 0.5, "lower": 0, "upper": 0.82, "floating": True},
                "alphal": {"name": "alphal", "value": 1.13, "lower": 0.1, "upper": 2, "floating": True},
                "alphar": {"name": "alphar", "value": 0.5, "lower": 0.3, "upper": 1.2, "floating": True},
                "nl": {"name": "nl", "value": 10, "lower": 1, "upper": 15, "floating": True},
                "nr": {"name": "nr", "value": 1, "lower": 1, "upper": 7, "floating": True}
            }}}

        obs = utilities.set_obs("AngleAccuracy", (limits[0], limits[1]))

        utilities.clear_existing_parameters()
        model = Models.Model(obs=obs, modelstring="Model", models=params_dscb)
        fit = Fitter.Fitter(array, model, retry=20)
        fit.fit()
        return fit.model.dict, fit.model.model.pdf, fit.gof


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
        + f"{text}\n",
        transform=ax.transAxes,
        fontsize=12)


if __name__ == "__main__":
    ta_distance = 30
    data = np.fromfile(
        f"/work/kschmidt/DeepJetCore/TrackReco_DeepJetCore/HGCalML/nntr_models/normal_detector/ta_distance_sweep/tad_{ta_distance}/Predicted/AngleAccuracy.csv")

    fitdict, pdf, gof = FitDSCB.dofit(data)
    with open(f"/work/kschmidt/DeepJetCore/TrackReco_DeepJetCore/HGCalML/nntr_models/normal_detector/ta_distance_sweep/tad_{ta_distance}/Predicted/AngleAccuracyFitDict", "wb") as file:
        pickle.dump(fitdict, file)

    print(fitdict)

    # Plotting

    limits = (-10, 10)
    bins = np.linspace(limits[0], limits[1], 128+1)
    x = np.linspace(limits[0], limits[1], 2000)
    y = pdf(x)
    params = fitdict["modelparameter"]["Model"]["parameter"]

    parameter = {}
    fit_parameter_names_dscb = ["mu", "sigma", "alphal", "alphar", "nl", "nr"]
    fit_parameter_names_cauchy = ["m", "gamma"]

    for key in fit_parameter_names_dscb:
        parameter[key] = unc.ufloat(
            params[key]["value"],
            abs(params[key]["value"] -
                max(abs(params[key]["lower"]), params[key]["upper"]))
        )

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
    # histogram
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
    plt.hlines(fwhm_height, x1, x2, color="red", label=f"FWHM={fwhm:.2f}°")
    # Fit
    label = "DSCB Fit Parameters:\n"
    for key in parameter.keys():
        label = label+f" {key}={parameter[key]:.2f}\n"
    label = label.rstrip("\n")
    plt.plot(
        x,
        y*density,
        label=label,
    )
    print("Goodness of fit:", gof)
    plt.xlim(bins[0], bins[-1])
    plt.xlabel(r"$\theta^{true} - \theta^{pred}$"+"[°]")
    plt.ylabel(f"Counts / ({(bins[1]-bins[0]):.1f} [°])")

    plt.legend()
    plt.savefig(f"/work/kschmidt/DeepJetCore/TrackReco_DeepJetCore/HGCalML/nntr_models/normal_detector/ta_distance_sweep/tad_{ta_distance}/Plots/angle_accuracy")
    plt.show()
