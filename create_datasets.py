import networkx as nx
from graph import Graph
import pandas as pd
import pickle as pkl
import os
from networkx import convert_node_labels_to_integers

def make_graph(file_path):
    lines = [line.rstrip('\n') for line in open(f'real_data/{file_path}.col')]

    g = nx.Graph()
    for l in lines:
        if len(l) > 1 and l[0] == 'e':
            sp = str.split(l, ' ')
            try:
                g.add_edge(int(sp[1])-1, int(sp[2])-1)
            except:
                g.add_edge(int(sp[3])-1, int(sp[-1])-1)

    obj = Graph('erdos_renyi', 1, 1, 0.1)
    g_ordered = convert_node_labels_to_integers(g)
    obj.g = g_ordered

    return obj

xl_file = pd.ExcelFile('google_metadata.xlsx')
clfs = {sheet_name: xl_file.parse(sheet_name) for sheet_name in xl_file.sheet_names}
available_files = os.listdir('real_data')
available_files = [a[:-4] for a in available_files]

diff_names = {}
solns = {}
sizes = {}
graphs = {}
for k in clfs.keys():
    diff_names[k] = list(clfs[k].Name)
    diff_solns = dict(zip(clfs[k].Name, clfs[k]['X(G)']))
    diff_sizes = dict(zip(clfs[k].Name, clfs[k]['|V|']))
    solns.update(diff_solns)
    sizes.update(diff_sizes)

    # generate networkx graphs
    diff_graphs = {}
    name_graphs = {}
    for n in range(len(diff_names[k])):
        if diff_names[k][n] in available_files:
            diff_graphs[n] = make_graph(diff_names[k][n])
            name_graphs[diff_names[k][n]] = diff_graphs[n]

    pkl.dump(diff_graphs, open(f'{k}.pkl', 'wb'))
    graphs.update(name_graphs)

pkl.dump(diff_names, open(f'{k}_names.pkl', 'wb'))
pkl.dump(solns, open(f'{k}_chromnum.pkl', 'wb'))
pkl.dump(sizes, open(f'{k}_vertexsizes.pkl', 'wb'))
pkl.dump(graphs, open(f'{k}_graphs.pkl', 'wb'))

# make a dataset with all real world graphs with size < 100
small_names = [k for k, v in sizes.items() if v < 100]
small_graphs = {}
named_graphs = {}
for n in range(len(small_names)):
    small_graphs[n] = graphs[small_names[n]]
    named_graphs[small_names[n]] = n

pkl.dump(small_graphs, open('small_graphs.pkl', 'wb'))
pkl.dump(named_graphs, open('named_graphs.pkl', 'wb'))


