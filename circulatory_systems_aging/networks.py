import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy


def get_connected_line_network(n):
    adj_mat = np.zeros((n,n))
    for i in range(0,n-1):
        adj_mat[i,i+1] = 1
    return adj_mat


def get_central_node_network(n):
    adjmat = np.zeros((n,n))
    # connect first node to all others (but not itself)
    adjmat[0,:] = 1
    adjmat[0,0] = 0
    return adjmat


def get_branching_network(p):
    n_nodes = 2**(p+1) - 1
    adjmat = np.zeros((n_nodes,n_nodes))
    for i in range(n_nodes):
        if i == 0:
            adjmat[i,1] = 1
            adjmat[i,2] = 1
        elif (i % 2) == 0:
            if (2*i + 2) >= n_nodes:
                break
            adjmat[i, 2*i] = 1
            adjmat[i, 2*i + 2] =  1
        else:
            if (2*i + 3) >= n_nodes:
                break
            adjmat[i, 2*i + 1] = 1
            adjmat[i, 2*i + 3] =  1
    return adjmat


def get_fully_connected_random_directional_network(n):
    adjmat = np.zeros((n,n))
    for i in range(n):
        for j in range(i):
            r = np.random.rand()
            if r < 0.5:
                adjmat[i,j] = 1
            else:
                adjmat[j,i] = 1
    return adjmat


def get_erdos_reyni_random_network(n,p):
    adjmat = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if i != j: # no self-edges
                r = np.random.rand()
                if r < p:
                    adjmat[i,j] = 1
    return adjmat


def get_barabasi_albert_network(n,m):
    graph = nx.barabasi_albert_graph(n,m)
    a = nx.adjacency_matrix(graph).todense()
    # make directed...
    for i in range(n):
        for j in range(i):
            if (a[i,j] == 1) and (a[j,i] == 1):
                r = np.random.rand()
                if r < 0.5:
                    a[i,j] = 0
                else:
                    a[j,i] = 0
    return a


# TODO: vis directed network..
def vis_network(ax, adjmat):
    graph = nx.Graph()
    n = adjmat.shape[0]
    for i in range(n):
        for j in range(n):
            if adjmat[i,j] == 1:
                graph.add_edge(i,j,weight=1)
    ax=nx.draw_networkx(graph, ax=ax)
#     pos = nx.spring_layout(graph)
#     ax=nx.draw_networkx_nodes(graph, pos, ax=ax)
#     ax=nx.draw_networkx_edges(graph, pos, ax=ax, arrows=True)
    return ax
