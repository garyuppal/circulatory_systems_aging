from network_model import NetworkAgingTorch
from utils import pickle_save, get_device
import networks as nwks
import numpy as np
import torch
import time
from pathlib import Path


def get_adj_mats():
    n = 1000
    p = 9 # gives 2**(9+1) - 1 = 1023 nodes

    prob_low = 0.1
    prob_high = 0.5
    m_low = 2
    m_high = 50

    # adjmats = {
    #     'hub': nwks.get_central_node_network(n),
    #     'line': nwks.get_connected_line_network(n),
    #     'branching': nwks.get_branching_network(p),
    #     f'ER_p{prob_low}': nwks.get_erdos_reyni_random_network(n, prob_low),
    #     f'ER_p{prob_high}': nwks.get_erdos_reyni_random_network(n, prob_high),
    #     f'BA_m{m_low}': nwks.get_barabasi_albert_network(n, m_low),
    #     f'BA_m{m_high}': nwks.get_barabasi_albert_network(n, m_high),
    #     'full': nwks.get_fully_connected_random_directional_network(n)
    # }
    adjmats = {
        'hub': nwks.get_central_node_network(n),
        'line': nwks.get_connected_line_network(n),
        'branching': nwks.get_branching_network(p),
        # f'ER_p{prob_low}': nwks.get_erdos_reyni_random_network(n, prob_low),
        # f'ER_p{prob_high}': nwks.get_erdos_reyni_random_network(n, prob_high),
        # f'BA_m{m_low}': nwks.get_barabasi_albert_network(n, m_low),
        # f'BA_m{m_high}': nwks.get_barabasi_albert_network(n, m_high),
        # 'full': nwks.get_fully_connected_random_directional_network(n)
    }
    return adjmats


def main(outpath):
    st = time.time()
    device = get_device()
    npop = 10000
    ntime = 1000
    pi_d = 0.002

    # varying cases
    repair_probs = [1e-4, 1e-3]
    init_dead_probs = [0, 0.1, 0.4]
    adj_matrices = get_adj_mats()
    mkeys = list(adj_matrices.keys())

    for pi_r in repair_probs:
        for eta in init_dead_probs:
            for k in mkeys:
                st_local = time.time()
                print(f"Running: pi_r={pi_r}, eta={eta}, topology={k}...")
                adjmat = adj_matrices[k]
                net = NetworkAgingTorch(adjmat, npop, eta, pi_d, pi_r, device)
                _, _ = net.age_network(ntime, verbose=100)
                print("saving...")
                pickle_save(outpath / f"{k}_ETA{eta}_PIR{pi_r}.pkl", net)
                sp_local = time.time()
                print(f"done: time elapsed = {sp_local-st_local}")
    sp = time.time()
    print(f"Total time: {sp-st}")
    print("***ALL DONE***")


if __name__ == "__main__":
    basepath = Path("./")
    outpath = basepath / "experiments" / "network_aging"
    outpath.mkdir(exist_ok=True, parents=True)
    main(outpath)
