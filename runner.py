"""
This is the machinnery that runs your agent in an environment.

This is not intented to be modified during the practical.
"""
import matplotlib.pyplot as plt
import numpy as np
import agent
import pickle as pkl
from tqdm import tqdm
import os

class Runner:
    def __init__(self, environment, agent, ckpt, verbose=False):
        self.environment = environment
        self.agent = agent
        self.verbose = verbose
        self.ckpt = ckpt

        if not os.path.exists(f'{ckpt}/epoch_data'):
            os.mkdir(f'{ckpt}/epoch_data')

    def step(self, train=True):
        observation = self.environment.observe().clone()
        action = self.agent.act(observation).copy()
        (reward, done) = self.environment.act(action)
        self.agent.reward(observation, action, reward,done, train)
        return (observation, action, reward, done)

    def loop(self, games, max_iter):
        avg_ratios = []
        avg_test = []
        avg_colors_train = []
        avg_colors_test = []
        ep = 0
        pkl.dump([self.environment.train_avg], open(f'{self.ckpt}/rnd_colors_train', 'wb'))
        pkl.dump([self.environment.test_avg], open(f'{self.ckpt}/rnd_colors_test', 'wb'))
        while True:
            print(f'epoch {ep}')
            self.agent.model.train()
            num_colors_train = []
            num_colors_test = []
            approx_ratios = []
            graphs = np.random.permutation(np.arange(games))
            for i in tqdm(range(games)): # games
                g = graphs[i]
                self.environment.reset(g)
                self.agent.reset(g)

                for i in range(max_iter):
                    # if self.verbose:
                    # print("Simulation step {}:".format(i))
                    (obs, act, rew, done) = self.step()
                    if done:
                        rnd_color = self.environment.soln
                        aprat = 1.0 * self.environment.num_colors / rnd_color
                        approx_ratios.append(aprat)
                        num_colors_train.append(self.environment.num_colors)
                        break

            avg_approx = np.mean(approx_ratios)
            avg_ratios.append(avg_approx)
            
            # run test set
            self.agent.model.eval()
            test_ratios = []
            test_graphs = np.random.choice(np.arange(100), size=20)
            test_colors = []
            for g in tqdm(range(20)):
                gr = test_graphs[g]
                self.environment.reset(gr, test=True)
                self.agent.reset(gr, test=True)
                
                for i in range(max_iter):
                    # if self.verbose:
                    # print("Simulation step {}:".format(i))
                    (obs, act, rew, done) = self.step(train=False)
                    if done:
                        rnd_color = self.environment.soln
                        aprat = 1.0 * self.environment.num_colors / rnd_color
                        test_ratios.append(aprat)
                        num_colors_test.append(self.environment.num_colors)
                        test_colors.append(rnd_color)
                        break

            a_test = np.mean(test_ratios)
            avg_test.append(a_test)
            avg_colors_train.append(np.mean(num_colors_train))
            avg_colors_test.append(np.mean(num_colors_test))
            
            pkl.dump(avg_ratios, open(f'{self.ckpt}/epoch_data/train_{ep}', 'wb'))
            pkl.dump(test_ratios, open(f'{self.ckpt}/epoch_data/test_{ep}', 'wb'))

            print(f'train: {np.mean(avg_ratios)}, test: {np.mean(test_ratios)}')
            print(f'train: ({avg_colors_train[-1]}, {self.environment.train_avg}), test: ({avg_colors_test[-1]}, {np.mean(test_colors)})')

            if ep % 5 == 0:
                self.agent.save_model(ep)
                pkl.dump(avg_ratios, open(f'{self.ckpt}/train_losses', 'wb'))
                pkl.dump(avg_test, open(f'{self.ckpt}/test_losses', 'wb'))
                pkl.dump(avg_colors_train, open(f'{self.ckpt}/train_colors', 'wb'))
                pkl.dump(avg_colors_test, open(f'{self.ckpt}/test_colors', 'wb'))

            ep += 1

        return 1

def iter_or_loopcall(o, count):
    if callable(o):
        return [ o() for _ in range(count) ]
    else:
        # must be iterable
        return list(iter(o))

def listify(o, count):
    return [o for _ in range(count)]

class BatchRunner:
    """
    Runs several instances of the same RL problem in parallel
    and aggregates the results.
    """

    def __init__(self, env_maker, agent_maker, count, verbose=False):
        self.environments = listify(env_maker, count)
        self.agents = listify(agent_maker, count)
        assert(len(self.agents) == len(self.environments))
        self.verbose = verbose
        self.ended = [ False for _ in self.environments ]

    def game(self, max_iter, g):
        rewards = []
        for (agent, env) in zip(self.agents, self.environments):
            env.reset(g)
            agent.reset(g)
            game_reward = 0
            for i in range(1, max_iter+1):
                observation = env.observe()
                action = agent.act(observation)
                (reward, stop) = env.act(action)
                agent.reward(observation, action, reward)
                game_reward += reward
                if stop =="stop":
                    break
            rewards.append(game_reward)
        return sum(rewards)/len(rewards)

    def loop(self, games, max_iter):
        cum_avg_reward = 0.0
        for g in range(1, games+1):
            avg_reward = self.game(max_iter, g)
            cum_avg_reward += avg_reward
            if self.verbose:
                print("Simulation game {}:".format(g))
                print(" ->            average reward: {}".format(avg_reward))
                print(" -> cumulative average reward: {}".format(cum_avg_reward))
        return cum_avg_reward
