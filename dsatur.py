import numpy as np
import pickle as pkl
import argparse
import networkx as nx

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_graphs', default='er_50_train', type=str,
                        help='path to graphs to compute soln for, pkl format')
    parser.add_argument('--out_app', default='_dsatur', type=str,
                        help='path to output file, appended to input graph file name')

    return parser.parse_args()

def dsatur(g):
    d = nx.algorithms.coloring.greedy_color(g, strategy=nx.coloring.strategy_saturation_largest_first)
    num_colors = max(d.values()) + 1 # since colors are 0-indexed
    return num_colors

def main():
    hps = parse_args()

    graphs = pkl.load(open(hps.input_graphs, 'rb'))
    solns = {}
    for k in graphs.keys():
        g = graphs[k].g
        solns[k] = dsatur(g)

    pkl.dump(solns, open(f'{hps.input_graphs}{hps.out_app}', 'wb'))
    
if __name__ == '__main__':
    main()
