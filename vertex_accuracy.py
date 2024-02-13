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
                "mu": {"name": "mu", "value": 0, "lower": -3, "upper": 3, "floating": True},
                "sigma": {"name": "sigma", "value": 95, "lower": 90, "upper": 115, "floating": True},
                "alphal": {"name": "alphal", "value": 1.13, "lower": 0.8, "upper": 1.15, "floating": True},
                "alphar": {"name": "alphar", "value": 1.1, "lower": 0.8, "upper": 1.6, "floating": True},
                "nl": {"name": "nl", "value": 13, "lower": 10, "upper": 15, "floating": True},
                "nr": {"name": "nr", "value": 6.18, "lower": 6, "upper": 10, "floating": True}
            }}}
        params_cauchy = {"Model": {
            "pdf": "Cauchy", "parameter": {
                "m": {"name": "m", "value": -5, "lower": -10, "upper": 10},
                "gamma": {"name": "gamma", "value": 300, "lower": 100, "upper": 300}
            }}}

        obs = utilities.set_obs("VertexAccuracy", (limits[0], limits[1]))

        utilities.clear_existing_parameters()
        model = Models.Model(obs=obs, modelstring="Model", models=params_dscb)
        fit = Fitter.Fitter(array, model, retry=20)
        fit.fit()
        return fit.model.to_dict(), fit.model.model.pdf


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
    data = pd.read_csv(
        f"/work/kschmidt/DeepJetCore/TrackReco_DeepJetCore/HGCalML/nntr_models/normal_detector/ta_distance_sweep/tad_{ta_distance}/Predicted/VertexAccuracy.csv").to_numpy()[:, 1]

    fitdict, pdf = FitDSCB.dofit(data)
    with open(f"/work/kschmidt/DeepJetCore/TrackReco_DeepJetCore/HGCalML/nntr_models/normal_detector/ta_distance_sweep/tad_{ta_distance}/Predicted/VertexAccuracyFitDict", "wb") as file:
        pickle.dump(fitdict, file)

    print(fitdict)

    # Plotting

    limits = (-800, 800)
    bins = np.linspace(limits[0], limits[1], 128+1)
    x = np.linspace(limits[0], limits[1], 1000)
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
    h = plt.hist(
        data,
        bins,
        label="MC Data",
        histtype="step",
        color="black",
        density=True
        )
    fwhm, fwhm_height, x1, x2 = FWHM(x, y/np.trapz(y, x))
    plt.hlines(fwhm_height, x1, x2, color="red", label=f"FWHM={fwhm:.2f} mm")
    label = "DSCB Fit Parameters:\n"
    for key in parameter.keys():
        label = label+f" {key}={parameter[key]:.2f}\n"
    plt.plot(
        x,
        y/np.trapz(y, x),
        label=label,
        )
    plt.xlim(bins[0], bins[-1])
    plt.xlabel(r"$V_{1,z}^{true} - V_{1,z}^{pred}$"+"[mm]")
    plt.ylabel(f"Counts / ({(bins[1]-bins[0]):.1f} [mm])")

    plt.legend()
    plt.savefig(f"/work/kschmidt/DeepJetCore/TrackReco_DeepJetCore/HGCalML/nntr_models/normal_detector/ta_distance_sweep/tad_{ta_distance}/Plots/vertex_accuracy")
    plt.show()
