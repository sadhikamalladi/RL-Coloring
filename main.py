import argparse
import agent
import environment
import runner
import graph
import logging
import numpy as np
import networkx as nx
import sys
import glob
import re
import pickle as pkl
import datetime, os

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
parser.add_argument('--ngames', type=int, metavar='n', default='100', help='number of games to simulate')
parser.add_argument('--niter', type=int, metavar='n', default='1000', help='max number of iterations per game')
parser.add_argument('--batch', type=int, metavar='nagent', default=None, help='batch run several agent at the same time')
parser.add_argument('--verbose', action='store_true', default=True, help='Display cumulative results at each step')
parser.add_argument('--interactive', action='store_true', help='After training, play once in interactive mode. Ignored in batch mode.')
parser.add_argument('--train_file', default='test', type=str,
                    help='loads data from pickle file if not None')
parser.add_argument('--test_file', default='train', type=str,
                    help='loads data from pickle file for testing if not None')
parser.add_argument('--ckpt', default=None, type=str,
                    help='directory to save results in, makes a new one using current time if None')
parser.add_argument('--load_model', default=False, action='store_true',
                    help='when not none loads model from checkpoint')

parser.add_argument('--test_only', default=False, action='store_true',
                    help='tests a model without training, load_model must be specified')
parser.add_argument('--test_output', default='testing_results.pkl', type=str,
                    help='output file for test results, used in test_only option')

def main():
    args = parser.parse_args()

    if args.train_file is None:
        graph_dic = {}
        for graph_ in range(args.graph_nbr):
            seed = np.random.seed(120+graph_)
            graph_dic[graph_]=graph.Graph(graph_type=args.graph_type, cur_n=100, p=0.14,m=4,seed=seed)
    else:
        graph_dic = pkl.load(open(args.train_file, 'rb'))

    args.ngames = len(list(graph_dic.keys()))


    current_ep = 0
    if args.load_model:
        # find epoch we were last at
        ckpts = glob.glob(f'{args.ckpt}/model_ckpts/*.pt')
        max_c = 0
        for c in ckpts:
            num = int(re.search(f'{args.ckpt}/model_ckpts/(.+?).pt', c).group(1))
            if num > max_c:
                max_c = num

        args.load_model = f'{args.ckpt}/model_ckpts/{max_c}.pt'
        current_ep = max_c

    if args.ckpt is None:
        now = datetime.datetime.today()
        args.ckpt = now.strftime('%m-%d-%y_%H.%M.%S')

    if args.test_only:
        print('testing only!')
        env_class = environment.Environment(graph_dic,args)
        
        agent_class = agent.Agent(graph_dic, args)
        my_runner = runner.Runner(env_class, agent_class, args.ckpt, args, args.verbose)
        my_runner.test(args.niter)
        return
        
    if not os.path.exists(args.ckpt):
        os.mkdir(args.ckpt)

    pkl.dump(args, open(f'{args.ckpt}/hps.pkl', 'wb'))


    logging.info('Loading agent...')
    agent_class = agent.Agent(graph_dic, args)

    logging.info('Loading environment %s' % args.environment_name)
    env_class = environment.Environment(graph_dic,args)

    if args.batch is not None:
        print("Running a batched simulation with {} agents in parallel...".format(args.batch))
        my_runner = runner.BatchRunner(env_class, agent_class, args.batch, args.ckpt,args, args.verbose)
        final_reward = my_runner.loop(args.ngames, args.niter)
        print("Obtained a final average reward of {}".format(final_reward))
        agent_class.save_model()
    else:
        print("Running a single instance simulation...")
        my_runner = runner.Runner(env_class, agent_class, args.ckpt, args, args.verbose)
        final_reward = my_runner.loop(args.ngames, args.niter, ep=current_ep)
        print("Obtained a final reward of {}".format(final_reward))
        agent_class.save_model()



if __name__ == "__main__":
    main()
