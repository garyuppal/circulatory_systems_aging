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
To reproduce the four figures in the paper, follow the steps below. Each figure has...

Model given config files, with overriding parameters for those we vary...

### Network aging analysis (figure 1)
Run the model simulations for each type of network analyzed in the paper:
```
bash run_figure1_sims.sh
```
Plot the results using the `plot_figure1.ipynb` notebook.

### Circulatory flow model analysis (figure 2)
Run the model simulations for each type of network analyzed in the paper:
```
bash run_figure2_sims.sh
```
Plot the results using the `plot_figure2.ipynb` notebook.

### Circulatory flow model analysis (figure 3)
Run the model simulations for each type of network analyzed in the paper:
```
bash run_figure3_sims.sh
```
Plot the results using the `plot_figure3.ipynb` notebook.

### Fitting to empirical data (figure 4)
