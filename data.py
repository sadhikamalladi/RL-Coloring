import numpy as np
import networkx as nx
import argparse
from graph import Graph
from tqdm import tqdm
import pickle as pkl

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_file', default='train', type=str,
                        help='dir to store generated graph data in')
    parser.add_argument('--graph_type', default='erdos_renyi', type=str,
                        help='type of graph to generate')
    parser.add_argument('--nbr_graphs', default=100, type=int,
                        help='number of graphs to generate')
    parser.add_argument('--nbr_nodes', default=100, type=int,
                        help='number of nodes in graph')
    parser.add_argument('--edge_prob', default=0.9, type=float,
                        help='probability of forming edge in graph')
    parser.add_argument('--edge_num', default=10, type=int,
                        help='number of edges in graph (used for barabasi and powerlaw')

    return parser.parse_args()

def main():
    hps = parse_args()

    graphs = {}
    for i in tqdm(range(hps.nbr_graphs)):
        g = Graph(hps.graph_type, hps.nbr_nodes, hps.edge_prob, m=hps.edge_num)
        graphs[i] = g

    pkl.dump(graphs, open(hps.data_file, 'wb'))

main()
