import torch
import argparse
from data import ColoringDataset, compute_num_nodes
from trainer import Trainer
from model import CombinatorialRL
import os
import glob
import re
import pickle as pkl

def parse_args():
    parser = argparse.ArgumentParser()

    # data params
    parser.add_argument('--train_file', type=str, default='train',
                        help='pickle file with training dataset')
    parser.add_argument('--test_file', type=str, default='test',
                        help='pickle file with testing dataset')

    # model params
    parser.add_argument('--embedding_size', type=int, default=128,
                        help='dimensionality of graph embedding')
    parser.add_argument('--hidden_size', type=int, default=128,
                        help='dimensionality of hidden layers')
    parser.add_argument('--n_glimpses', type=int, default=1,
                        help='number of glimpses back (uses attn)')
    parser.add_argument('--tanh_exploration', type=int, default=10,
                        help='context for pointer attention')
    parser.add_argument('--use_tanh', default=True, type=bool,
                        help='use tanh for pointer attention context')
    parser.add_argument('--attention_type', default='Dot', type=str,
                        help='type of attention mechanism from {Dot, Bahdanau}')

    # training params
    parser.add_argument('--beta', type=float, default=0.9,
                        help='beta for moving avg on critic')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch size')
    parser.add_argument('--epochs', type=int, default=5,
                        help='number of epochs to train')
    parser.add_argument('--grad_norm', type=float, default=2.0,
                        help='norm to clip grads at')

    # model saving/loading params
    parser.add_argument('--ckpt', type=str, default='output',
                        help='directory to save results and model ckpts in')
    parser.add_argument('--resume', type=bool, default=False,
                        help='when true, resumes model training from last epoch in ckpt folder')

    return parser.parse_args()

def main():
    hps = parse_args()

    if not os.path.exists(hps.ckpt):
        os.mkdir(hps.ckpt) 

    dsets = {'train': ColoringDataset(hps.train_file),
             'test': ColoringDataset(hps.test_file)}

    pkl.dump(dsets['train'].avg_rnd, open(f'{hps.ckpt}/avg_rnd_train.pkl', 'wb'))
    pkl.dump(dsets['test'].avg_rnd, open(f'{hps.ckpt}/avg_rnd_test.pkl', 'wb'))

    num_nodes = compute_num_nodes(dsets['train'])
    model = CombinatorialRL(hps, num_nodes)

    if hps.resume:
        ckpts = glob.glob(f'{hps.ckpt}/model*pt')
        max_c = 0
        for c in ckpts:
            num = int(re.search(f'{hps.ckpt}/model_(.+?).pt', c).group(1))
            if num > max_c:
                max_c = num
        model.load_state_dict(torch.load(f'{hps.ckpt}/model_{max_c}.pt'))
    
    trainer = Trainer(hps, dsets, model)
    trainer.train()

if __name__ == '__main__':
    main()
