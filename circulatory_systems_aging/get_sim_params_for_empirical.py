from utils import pickle_save, pickle_load, relu_type, fit_relu, calc_mort
import numpy as np
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.optimize import curve_fit


def main(args):
    base_simulation_path = Path(args.base_simulation_path)
    empirical_fit_path = Path(args.empirical_fit_path)
    outpath = Path(args.outpath)
    outpath.mkdir(parents=True, exist_ok=True)

    kvals = 1,2,3
    gammavals = 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0
    # 0, 0.25, 0.5, 0.75, 1.0

    nk = len(kvals)
    ng = len(gammavals)

    x_values = []
    y_values = []
    k_values = []
    gamma_values = []

    for j,k in enumerate(kvals):
        for i,gamma in enumerate(gammavals):
            respath = base_simulation_path / f"K{k}_gamma_{gamma}"
            res = pickle_load(respath / "results.pkl")

            # x, mort, y = calc_mort(res, w=4)

            mort = res['mort']
            t = np.linspace(0,1,len(mort))
            x = t
            y = mort/np.nanmean(mort)
            # ax[j], x, y = plot_mortality2(ax[j],res)
            
            x_values.append(x)
            y_values.append(y)
            k_values.append(k)
            gamma_values.append(gamma)

            
    # fit data for each k and gamma value
    threshold_vals = []
    slope_vals = []

    for x_data, y_data in zip(x_values, y_values):
        xt, m = fit_relu(x_data, y_data)
        threshold_vals.append(xt)
        slope_vals.append(m)

    kv = np.reshape(k_values,(nk,ng))
    gv = np.reshape(gamma_values,(nk,ng))
    sv = np.reshape(slope_vals,(nk,ng))
    tv = np.reshape(threshold_vals,(nk,ng))

    pfits = []
    degree = 3
    fit_coeffs = []
    for i in range(nk):
        x = gv[i,:]
        y = tv[i,:]
        coeffs = np.polyfit(x,y, degree)
        fit_coeffs.append(coeffs)

        # Generate polynomial function from coefficients
        p = np.poly1d(coeffs)
        pfits.append(p)


    basefits = {'thresholds': tv, 'slopes': sv, 'kvals': kv, 'gammavals': gv, 'fit_coeffs': fit_coeffs, 'pfits': pfits}
    # save the fits
    pickle_save(empirical_fit_path / "basefits.pkl", basefits)

    # fig, ax = plt.subplots()
    # for i in range(3):
    #     ax.scatter(gv[i,:],tv[i,:],label=f'k={kv[i,0]}')
    #     x_fit = np.linspace(0,1,100)
    #     y_fit = pfits[i](x_fit)/
    #     ax.plot(x_fit, y_fit, label=f'k={kv[i,0]}')
    #     ax.set_ylim(0,1)
    # ax.legend()
    # plt.show()

    #! load empirical fits [from relu fits given in another script...]
    fitted_thresholds = pickle_load(empirical_fit_path / "fitted_thresholds.pkl")
    fitted_slopes = pickle_load(empirical_fit_path / "fitted_slopes.pkl")
    mort_data = pickle_load(empirical_fit_path / "mort_data.pkl")
    
    #! get target for each species threshold value
    candidates = [
        "Homo sapiens (Japan2009Female)", 
        "Poecillia reticulata", 
        "Fulmarus glacialoides", 
        "Orcinus orca", 
        "C. elegans N2", #"Caenorhabditis elegans", 
        "Ceratitis capitata", 
        "Pinus sylvestris"] #,  
        # "Lacerta vivipara"
        # ]

    fitgammas = {} # each one a list of length 3 (for each K value; na's for when unique fit not found...)
    for species in candidates:
        target = fitted_thresholds[species]
        fgamma = []
        for i in range(3):
            cfit = np.copy(fit_coeffs[i])
            cfit[-1] -= target
            # solve
            roots = np.roots(cfit)
            # Filter real roots (since numpy.roots might return complex values)
            real_roots = roots[np.isreal(roots)].real

            if len(real_roots) == 1:
                # check if value between 0 and 1
                if (real_roots[0] < 0) or (real_roots[0] > 1):
                    fgamma.append(np.nan)
                else:
                    fgamma.append(real_roots[0])
            else:
                fgamma.append(np.nan)
        fitgammas[species] = fgamma

    pickle_save(outpath / "empirical_fitgammas.pkl", fitgammas)

    # save as csv to run in script
    params = pd.DataFrame(fitgammas)
    params.index = params.index + 1
    params.index.name = "K"
    params.reset_index(inplace=True)
    df = pd.melt(params, id_vars=["K"], var_name="Species", value_name="Gamma")
    df['Gamma'] = df['Gamma'].map(lambda x: f"{x:.5f}" if pd.notna(x) else "-1")
    df.to_csv(outpath / "empirical_fitgammas.csv", index=False)
    print("**ALL DONE**")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Get best model parameters for empirical mortality data.")
    parser.add_argument('--empirical_fit_path', type=str, dest="empirical_fit_path", help='Path to the empirical ramp function fit results.')
    parser.add_argument('--base_simulation_path', type=str, dest="base_simulation_path", help='Path to base simulation results.')
    parser.add_argument('--outpath', type=str, dest="outpath", help='Path to save the output files.')
    args = parser.parse_args()
    main(args)
