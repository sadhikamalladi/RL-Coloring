import torch
from torch.utils.data import DataLoader
import math
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from tqdm import trange
import pickle as pkl
USE_CUDA=True

class Trainer:

    def __init__(self, hps, dsets, model):
        self.hps = hps
        self.dsets = dsets
        self.model = model
        if USE_CUDA:
            self.model = self.model.cuda()

        if hps.test_only:
            hps.batch_size = 1
        self.train_loader = DataLoader(dsets['train'],
                                       batch_size=hps.batch_size,
                                       shuffle=True,
                                       num_workers=1)
        self.test_loader = DataLoader(dsets['test'],
                                      batch_size=hps.batch_size,
                                      shuffle=True,
                                      num_workers=1)

        self.actor_optim = torch.optim.Adam(model.actor.parameters(), lr=hps.lr)

    def test(self):
        self.model.eval()
        test_colors = []
        rnd_colors = []
        for val_batch in self.test_loader:
            import pdb;pdb.set_trace()
            inputs, graphs, rnd_color = val_batch
            inputs = Variable(inputs)
            inputs = inputs.cuda()

            originals = []
            for g in graphs:
                originals.append(self.dsets['test'].original_graphs[g.item()])
            R, actions, probs, colors = self.model(inputs, originals)

            test_colors.extend(colors)
            rnd_colors.extend(rnd_color)

        import pdb;pdb.set_trace()

        test_col.append(np.mean(avg_col_test))

    def train(self):
        critic_exp_mvg_avg = torch.zeros(1)
        if USE_CUDA: 
            critic_exp_mvg_avg = critic_exp_mvg_avg.cuda()

        train_loss = []
        val_loss = []
        train_col = []
        test_col = []
        for epoch in range(self.hps.epochs):
            avg_col_train = []
            for batch_id, sample_batch in enumerate(self.train_loader):
                self.model.train()

                inputs, graphs, rnd_colors = sample_batch
                inputs = Variable(inputs)
                inputs = inputs.cuda()

                originals = []
                for g in graphs:
                    originals.append(self.dsets['train'].original_graphs[g.item()])

                R, actions, probs, colors = self.model(inputs, originals)
                avg_col_train.append(torch.mean(colors).item())

                if batch_id == 0:
                    critic_exp_mvg_avg = R.mean()
                else:
                    critic_exp_mvg_avg = (critic_exp_mvg_avg * self.hps.beta) + ((1. - self.hps.beta) * R.mean())


                advantage = R - critic_exp_mvg_avg

                logprobs = 0
                for prob in probs: 
                    logprob = torch.log(prob)
                    logprobs += logprob
                logprobs[logprobs < -1000] = 0.  

                reinforce = advantage * logprobs
                actor_loss = reinforce.mean()

                self.actor_optim.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm(self.model.actor.parameters(),
                                    float(self.hps.grad_norm), norm_type=2)

                self.actor_optim.step()

                critic_exp_mvg_avg = critic_exp_mvg_avg.detach()

                approx_ratio = colors / rnd_colors
                train_loss.append(torch.mean(approx_ratio).item())

            train_col.append(np.mean(avg_col_train))

            self.model.eval()
            avg_col_test = []
            for val_batch in self.test_loader:
                inputs, graphs, rnd_colors = val_batch
                inputs = Variable(inputs)
                inputs = inputs.cuda()
                
                originals = []
                for g in graphs:
                    originals.append(self.dsets['test'].original_graphs[g.item()])
                R, actions, probs, colors = self.model(inputs, originals)
                avg_col_test.append(torch.mean(colors).item())
                val_approx = colors / rnd_colors
                val_loss.append(torch.mean(val_approx).item())

            test_col.append(np.mean(avg_col_test))

            print(f'epoch {epoch}: train ({np.mean(train_loss)}), test ({np.mean(val_loss)})')
            print(f"epoch {epoch}: train ({train_col[-1]}, {self.dsets['train'].avg_rnd}), test ({test_col[-1]}, {self.dsets['test'].avg_rnd})")

            pkl.dump(train_loss, open(f'{self.hps.ckpt}/train_losses.pkl', 'wb'))
            pkl.dump(val_loss, open(f'{self.hps.ckpt}/test_losses.pkl', 'wb'))
            pkl.dump(train_col, open(f'{self.hps.ckpt}/train_colors.pkl', 'wb'))
            pkl.dump(test_col, open(f'{self.hps.ckpt}/test_colors.pkl', 'wb'))
            torch.save(self.model.state_dict(), f'{self.hps.ckpt}/model_{epoch}.pt')
