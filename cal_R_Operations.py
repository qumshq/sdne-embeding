from copy import deepcopy
import networkx as nx


def cal_R_nx(g, N):
    R = 0
    G_t = deepcopy(g)
    for atk in range(N):
        deg = [G_t.degree(i) for i in range(N)]
        atk_ind = deg.index(max(deg))
        '''Get current node with largest degree'''
        dele = []
        for i in G_t[atk_ind]:
            dele.append(i)
        for i in dele:
            G_t.remove_edge(i, atk_ind)
        largest_cc = max(nx.connected_components(G_t), key=len) # The largest connected component
        R = R + float(len(largest_cc))/N
    return R/N