import networkx as nx
import numpy as np


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


def get_directed_SFN(n,m):
    anet = np.zeros((n,n))
    # start with m nodes
    anet[:m,:m] = 1

    # zero diagonal
    for i in range(n):
        anet[i,i] = 0

    # for each new node, add m in and out connections based on degree
    for k in range(m,n):
        # use choice method
        indegrees = anet.sum(axis=0)
        outdegrees = anet.sum(axis=1)
        total = anet.sum()

        # depend on m preexisting nodes with prob outdegrees
        other_nodes = np.random.choice(k, size=m, replace=False, p=outdegrees[:k]/total)
        anet[other_nodes,k] = 1
        
        # others depend on k with prob indegrees
        other_nodes = np.random.choice(k, size=m, replace=False, p=indegrees[:k]/total)
        anet[k,other_nodes] = 1
    return anet


def get_adjacency_matrix(config):
    network_type = config["Network"]["type"]

    if network_type == 'line':
        return get_connected_line_network(config.getint('Network', 'num_nodes'))
    elif network_type == 'star':
        return get_central_node_network(config.getint('Network', 'num_nodes'))
    elif network_type == 'branching':
        return get_branching_network(config.getint('Network', 'branching_power'))
    elif network_type == 'full':
        return get_fully_connected_random_directional_network(config.getint('Network', 'num_nodes'))
    elif network_type == 'erdos_reyni':
        return get_erdos_reyni_random_network(config.getint('Network', 'num_nodes'), config.getfloat('Network', 'random_network_p'))
    elif network_type == 'directed_SFN':
        return get_directed_SFN(config.getint('Network', 'num_nodes'), config.getfloat('Network', 'scale_free_m'))
    else:
        raise ValueError(f"Unknown network type: {network_type}")


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
