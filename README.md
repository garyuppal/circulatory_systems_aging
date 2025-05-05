# Linking Transport Biophysics to Phylogenetic Patterns in Aging

This repository contains the computational models, simulation scripts, and analysis code accompanying the manuscript:
``Linking Transport Biophysics to Phylogenetic Patterns in Aging'' ***(citation to be added)***.


## Installation
Pull this code via
```
git clone https://github.com/garyuppal/circulatory_systems_aging.git
```
Ensure you have the appropriate dependencies installed. We recommend using a virtual environment. Install the required Python packages (not including PyTorch) with:
```
pip install -r requirements.txt
```
Next install [PyTorch](https://pytorch.org/) manually based on your system and whether you want GPU support:

#### Linux or Windows (with NVIDIA GPU and CUDA 11.8)
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### Linux or Windows (CPU only)
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

#### MacOS (CUDA not supported)
```
pip3 install torch torchvision torchaudio
```

## Running the models
This repository includes two main computational models of aging:

1. Network Aging Model
    
    Simulates damage propagation in abstract networks with varying topologies. Each node represents a functional unit (e.g., a cell or organ), and connections define dependency relationships. Over time, damage accumulates and may spread through connected nodes, allowing exploration of how network structure influences aging and system failure rates.

2. Circulatory Flow Model

    Simulates damage propagation in a circular network where cells communicate via cooperative (beneficial) or antagonistic (harmful) factors that spread (via diffusion and advection) and decay through the shared fluid.

### Model execution
Each model is configured via an configuration file containing parameter values such as network structure, interaction strength, decay rates, and simulation time.

Each model is run with a configuration file passed giving model paramters.

#### Running the Network Aging Model

To run the network model, use the following command:

```
python network_model.py --config <config_file.cfg>
```

#### Running the Circulatory Flow Model
To run the circulatory flow model, use:
```
python circulatory_flow_model.py --config <config_file.cfg>
```

## Running the analyses in the paper
To reproduce the figures presented in the paper, run the simulation scripts located in the `scripts/` directory and then generate the corresponding plots using the associated Jupyter notebooks.

Note, all scripts should be run from the scripts directory. This requirement is in place to match relative pathing of the root scripts/settings.sh environment file, from which all other paths are defined relatively.

Each script executes a model using predefined configuration files (in .cfg format), with certain parameters overridden directly on the command line. These overrides allow for systematic exploration of model behavior across a range of biologically relevant values.

Below are the instructions for each figure:

### Figure 1 - Network aging analysis
This figure explores how network topology influences aging dynamics.
1. Run the model simulations for each type of network analyzed in the paper:
```
bash run_figure1_sims.sh
```
2. Visualize the results: Open and run the `plot_figure1.ipynb` notebook.

### Figure 2 - Circulatory flow model: effects of hill constants $K$ and mixing factor $\gamma$
1. Run the model simulations:
```
bash run_figure2_sims.sh
```
2. Plot the results using the `plot_figure2.ipynb` notebook.

### Figure 3- Circulatory flow model: effects of diffusion and flow rate
1. Run the model simulations:
```
bash run_figure3_sims.sh
```
2. Plot the results using the `plot_figure3.ipynb` notebook.

### Figure 4 - Fitting the model to empirical data
1. Run the model over an initial range of base parameters
```
bash run_figure4_base_sims.sh
```
2. Fit ramp functions to the empirical data
```
bash fit_ramp_to_empirical.sh
```
3. Obtain fit parameters to run the model for each species in the empirical data
```
bash get_empirical_simulation_parameters.sh
```
4. Run the flow model with a final set of fit parameters
```
bash run_figure4_with_fit_parameters.sh
```
5. Plot the results using the `plot_figure4.ipynb` notebook.
