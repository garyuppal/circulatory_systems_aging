import torch
import pickle
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.signal import savgol_filter


def get_device():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return device


def pickle_save(file, data):
    with open(file, "wb") as h:
        pickle.dump(data, h)


def pickle_load(file):
    with open(file, "rb") as h:
        data = pickle.load(h)
    return data


def export_networkx_to_graphML(adjmat, file):
    graph = nx.DiGraph()
    n = adjmat.shape[0]
    for i in range(n):
        for j in range(n):
            if adjmat[i,j] == 1:
                graph.add_edge(i,j,weight=1)

    nx.write_graphml(graph, file)
    

def visualize_flow_simulation(state, oidx=0, ghosts=True):
    goods = state['goods']
    toxins = state['toxins']
    cells = state['cells'][:,oidx,:]
    if ghosts:
        gfield = goods[:,oidx,:]
        tfield = toxins[:,oidx,:]
    else:
        gfield = goods[:,oidx,1:-1]
        tfield = toxins[:,oidx,1:-1]
    max_field = max(np.max(gfield), np.max(tfield))

    xmin = 0
    xmax = 1
    if ghosts == False:
        nx = gfield.shape[1]
    else:
        nx = gfield.shape[1] - 2
    x = np.linspace(xmin,xmax,nx)
    dx = x[1] - x[0]
    if ghosts:
        xfield = np.linspace(xmin-dx, xmax+dx, nx+2)
    else:
        xfield = x

    fig, ax= plt.subplots()
    gline, = ax.plot(xfield, gfield[0,:])
    tline, = ax.plot(xfield, tfield[0,:])
    scat, = ax.plot(x[cells[0,:]>0.5], np.zeros_like(x[cells[0,:]>0.5]), 'o', color='black')
    ax.set_ylim(-0.05*max_field, 1.1*max_field)
    ax.set_xlim(np.amin(xfield), np.amax(xfield))

    def update(frame_idx):
        gline.set_ydata(gfield[frame_idx,:])
        tline.set_ydata(tfield[frame_idx,:])
        scat.set_xdata(x[cells[frame_idx,:]>0.5])
        scat.set_ydata(np.zeros_like(x[cells[frame_idx,:]>0.5]))
        ax.set_title(f"Time: {frame_idx}/{gfield.shape[0]}")
        return gline,
    ani = animation.FuncAnimation(fig, update, frames=gfield.shape[0], blit=False)
    plt.show()


def calc_mort(res, w=11, p=1):
    # TODO: use deltaT from std for w???
    pvt = res["pop_v_time"]
    times = res["times"]
    age_delay = res["age_delay"]

    nx = pvt.shape[1]
    alive_threshold = 0.01*nx

    surv = (pvt > alive_threshold).sum(axis=1)

    num_organisms = pvt[0,:].max()

    surv95_threshold = 0.05 * num_organisms
    surviving_indices = np.where(surv > surv95_threshold)[0]

    if surviving_indices.size > 0:
        t95 = times[surviving_indices[-1]] - age_delay  # Last timepoint with >5% survival
    else:
        t95 = np.inf  # If survival never drops below 5%, set to infinity

    astart = np.argmax(times > age_delay)  # Start index after aging begins
    at95 = np.argmax(times > (t95 + age_delay)) if np.isfinite(t95) else len(times)


    s95 = surv[astart:at95]

    dt = (times[1] - times[0])

    smoothed = savgol_filter(s95, window_length=w, polyorder=p)
    dsurvdt = savgol_filter(s95, window_length=w, polyorder=p, deriv=1, delta=dt)
    mort = -dsurvdt/smoothed

    normmortplot = mort / np.nanmean(mort)
    t = np.linspace(0,1, len(mort))
    return t, mort, normmortplot
