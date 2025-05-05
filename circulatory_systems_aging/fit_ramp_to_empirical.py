import numpy as np
import torch
import time
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
import pandas as pd
from utils import pickle_save, get_device, pickle_load, relu_type, fit_relu


def compute_mortality(data, qx=False, lx_column=None, Nx_column=None):
    """
    Compute qx (probability of death) and mux (hazard rate) from Nx (number alive) or lx (survivorship).
    
    Args:
        data (pd.DataFrame): Data containing Nx or lx.
        lx_column (str): Column name for lx (proportion surviving, optional).
        Nx_column (str): Column name for Nx (absolute count surviving, optional).
    
    Returns:
        pd.DataFrame: Original data with computed qx and mux.
    """
    
    if qx == False:
        if lx_column is not None:
            data['lx'] = data[lx_column]
        elif Nx_column is not None:
            data['lx'] = data[Nx_column] / data[Nx_column].iloc[0]  # Normalize to lx (0 to 1)
        else:
            raise ValueError("Must provide either lx_column or Nx_column.")

        # Compute qx (Probability of death)
        data['qx'] = np.nan
        print(f"SHAPE = {data['qx'].shape}")
    #     data.loc[:-1, 'qx'] = (data['lx'].values[:-1] - data['lx'].values[1:]) / data['lx'].values[:-1]
    #     data.loc[:(len(data['qx'])-2), 'qx'] = (data['lx'].values[:-1] - data['lx'].values[1:]) / data['lx'].values[:-1]
        data.iloc[:-1, data.columns.get_loc('qx')] = (data['lx'].values[:-1] - data['lx'].values[1:]) / data['lx'].values[:-1]

        # Ensure last value of qx is 1 (final death probability)
        data['qx'].iloc[-1] = 1.0
    
    # Compute mux (Hazard rate)
    data['mux'] = -np.log(1 - data['qx'])
    
    return data


def apply_survival_cutoff(data, lx_column='lx', cutoff=0.05):
    """
    Applies a cutoff at the age where survivorship (lx) drops below the specified threshold.
    
    Args:
        data (pd.DataFrame): Data containing an 'Age' column and 'lx' (survivorship).
        lx_column (str): Column name for survivorship values.
        cutoff (float): Threshold for survivorship cutoff (default: 5%).
    
    Returns:
        pd.DataFrame: Truncated data up to the cutoff age.
    """
    # Find the first age where lx <= cutoff
    cutoff_index = data[data[lx_column] <= cutoff].index.min()
    
    # Truncate the data at that index
    if not np.isnan(cutoff_index):
        data = data.loc[:cutoff_index]
    
    return data


def safe_mean(arr):
    """
    Computes the mean of a NumPy array while ignoring NaNs and Infs.
    
    Args:
        arr (np.ndarray): Input array.
    
    Returns:
        float: Mean of valid values, or NaN if no valid values exist.
    """
    # Mask out NaN and Inf values
    valid_values = arr[np.isfinite(arr)]  # Keeps only finite numbers
    
    if valid_values.size == 0:
        return np.nan  # Return NaN if all values are NaN/Inf
    
    return np.mean(valid_values)



def main(args):
    datapath = Path(args.datapath)
    outpath = Path(args.outpath)

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

    #! get data
    xemp = {}
    yemp = {}

    for species in candidates:
        data = pd.read_excel(datapath, sheet_name=species, header=1)
        print(species)
        
        t = data['x']
        if 'qx' in data.columns:
            data = compute_mortality(data, qx=True)
        elif 'lx' in data.columns:
            data = compute_mortality(data, lx_column='lx')
            print(f'computing mort from lx for {species}')
        elif 'Nx' in data.columns:
            data = compute_mortality(data, Nx_column='Nx')
            print(f'computing mort from Nx for {species}')
        else:
            print("NONE FOUND!")
        
    #     print(data)
        cutdata = apply_survival_cutoff(data)
    #     print(cutdata)
        t = cutdata['x']
        mort = cutdata['mux']
    #     fig, ax = plt.subplots()
    #     ax.plot(t,mort/safe_mean(mort),'-o')
        
        xemp[species] = t
        yemp[species] = mort/safe_mean(mort)
    #     ax.set_ylim(0,5)
    #     ax.set_title(species)

    #! plot fits
    #! save fits
    fitted_thresholds = {}
    fitted_slopes = {}
    mort_data = {}

    for species in candidates:
        print(species)
        
        x = xemp[species].values
        x_data = x/x[-1]
        y_data = yemp[species].values
        
        # remove infs/nans
        x_data = x_data[np.isfinite(y_data)]
        y_data = y_data[np.isfinite(y_data)]
        
        # fit relu
        fitted_x_threshold, fitted_slope = fit_relu(x_data,y_data)
        
        fitted_thresholds[species] = fitted_x_threshold
        fitted_slopes[species] = fitted_slope
        mort_data[species] = [x_data, y_data]
    #     # plot
    #     fig, ax = plt.subplots()
    #     ax.plot(x_data,y_data,'o')
    #     ax.plot(x_data, relu_type(x_data, fitted_x_threshold, fitted_slope), color='red', label="Fitted ReLU-type function")
    #     ax.axvline(fitted_x_threshold, color='green', linestyle='--', label="Fitted x_threshold")
    #     ax.legend()
    #     ax.set_xlabel("Time")
    #     ax.set_ylabel("Normalized mortality")

    #     #     ax.set_ylim(0,5)
    #     ax.set_title(species)
    # # plt.show()
        # plt.savefig(emailfigs / f"data_fit {species}.png", bbox_inches="tight")

    #! save fits
    outpath = outpath / "empirical_fits"
    outpath.mkdir(exist_ok=True, parents=True)

    pickle_save(outpath / "fitted_thresholds.pkl", fitted_thresholds)
    pickle_save(outpath / "fitted_slopes.pkl", fitted_slopes)
    pickle_save(outpath / "mort_data.pkl", mort_data)
    print("***ALL DONE***")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fit ReLU-type function to empirical mortality data.")
    parser.add_argument('--datapath', type=str, dest="datapath", help='Path to the data directory.')
    parser.add_argument('--outpath', type=str, dest="outpath", help='Path to save the output files.')
    args = parser.parse_args()
    main(args)
