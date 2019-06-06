import argparse
import numpy as np
import os
import pickle as pkl
from data import ColoringDataset

def parse_args():
    parser = argparse.ArgumentParser()

    # data params
    parser.add_argument('--train_file', type=str, default='train',
                        help='pickle file with training dataset')
    parser.add_argument('--test_file', type=str, default='test',
                        help='pickle file with testing dataset')

    # model params
    parser.add_argument('--reg_hidden', type=int, default=64,
                        help='number of hidden units in state embedding')
    parser.add_argument('--embed_dim', type=int, default=64,
                        help='number of dimensions to embed in')
    parser.add_argument('--msg_pass', type=int, default=5,
                        help='number of times to message pass in S2V')

    # training params

def main():
    hps = parse_args()

    if not os.path.exists(hps.ckpt):
        os.mkdir(hps.ckpt)

    dsets = {'train': ColoringDataset(hps.train_file),
             'test': ColoringDataset(hps.test_file)}
    
