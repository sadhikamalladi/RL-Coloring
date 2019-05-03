import argparse
import agent
import environment
import runner
import graph
import logging
import numpy as np
import networkx as nx
import sys
import pickle as pkl

# # 2to3 compatibility
# try:
#     input = raw_input
# except NameError:
#     pass

# Set up logger
logging.basicConfig(
    format='%(asctime)s:%(levelname)s:%(message)s',
    level=logging.INFO
)

parser = argparse.ArgumentParser(description='RL running machine')
parser.add_argument('--environment_name', metavar='ENV_CLASS', type=str, default='COLORING', help='Class to use for the environment. Must be in the \'environment\' module')
parser.add_argument('--agent', metavar='AGENT_CLASS', default='Agent', type=str, help='Class to use for the agent. Must be in the \'agent\' module.')
parser.add_argument('--graph_type',metavar='GRAPH', default='erdos_renyi',help ='Type of graph to optimize')
parser.add_argument('--graph_nbr', type=int, default='1000', help='number of graph to generate')
parser.add_argument('--model', type=str, default='S2V_QN_1', help='model name')
parser.add_argument('--ngames', type=int, metavar='n', default='500', help='number of games to simulate')
parser.add_argument('--niter', type=int, metavar='n', default='1000', help='max number of iterations per game')
parser.add_argument('--batch', type=int, metavar='nagent', default=None, help='batch run several agent at the same time')
parser.add_argument('--verbose', action='store_true', default=True, help='Display cumulative results at each step')
parser.add_argument('--interactive', action='store_true', help='After training, play once in interactive mode. Ignored in batch mode.')
parser.add_argument('--load_data', default=None, type=str,
                    help='loads data from npy file if not None')

def main():
    args = parser.parse_args()

    if args.load_data is None:
        graph_dic = {}
        for graph_ in range(args.graph_nbr):
            seed = np.random.seed(120+graph_)
            graph_dic[graph_]=graph.Graph(graph_type=args.graph_type, cur_n=100, p=0.14,m=4,seed=seed)
    else:
        graph_dic = pkl.load(open(args.load_data, 'rb'))
            

    logging.info('Loading agent...')
    agent_class = agent.Agent(graph_dic, args.model)

    logging.info('Loading environment %s' % args.environment_name)
    env_class = environment.Environment(graph_dic,args.environment_name)

    if args.batch is not None:
        print("Running a batched simulation with {} agents in parallel...".format(args.batch))
        my_runner = runner.BatchRunner(env_class, agent_class, args.batch, args.verbose)
        final_reward = my_runner.loop(args.ngames, args.niter)
        print("Obtained a final average reward of {}".format(final_reward))
        agent_class.save_model()
    else:
        print("Running a single instance simulation...")
        my_runner = runner.Runner(env_class, agent_class, args.verbose)
        final_reward = my_runner.loop(args.ngames, args.niter)
        print("Obtained a final reward of {}".format(final_reward))
        agent_class.save_model()



if __name__ == "__main__":
    main()
