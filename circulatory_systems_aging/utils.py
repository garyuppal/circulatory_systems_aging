import torch
import pickle
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation


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
