"""
This is the machinnery that runs your agent in an environment.

This is not intented to be modified during the practical.
"""
import matplotlib.pyplot as plt
import numpy as np
import agent

class Runner:
    def __init__(self, environment, agent, verbose=False):
        self.environment = environment
        self.agent = agent
        self.verbose = verbose

    def step(self):
        observation = self.environment.observe().clone()
        action = self.agent.act(observation).copy()
        (reward, done) = self.environment.act(action)
        self.agent.reward(observation, action, reward,done)
        return (observation, action, reward, done)

    def loop(self, games, max_iter):

        cumul_reward = 0.0
        list_cumul_reward_game=[]
        list_optimal_set = []
        list_aprox_set =[]
        mean_reward = []
        for epoch_ in range(25):
            print(f'Epoch {epoch_}')
            for g in range(games):
                self.environment.reset(g)
                self.agent.reset(g)
                cumul_reward_game = 0.0

                for i in range(max_iter):
                    # if self.verbose:
                    # print("Simulation step {}:".format(i))
                    (obs, act, rew, done) = self.step()
                    cumul_reward += rew
                    cumul_reward_game += rew
                           
                    if done:
                        if self.verbose:
                            print(f'Guessed colors: {self.environment.num_colors}, Random colors: {self.environment.get_optimal_sol()}')
                        break
                np.savetxt('test_'+str(epoch_)+'.out', list_optimal_set, delimiter=',')
                np.savetxt('test_approx_' + str(epoch_) + '.out', list_aprox_set, delimiter=',')

                #np.savetxt('opt_set.out', list_optimal_set, delimiter=',')

            if self.verbose:
                print(" <=> Finished game number: {} <=>".format(g))
                print("")

        np.savetxt('test.out', list_cumul_reward_game, delimiter=',')
        np.savetxt('opt_set.out', list_optimal_set, delimiter=',')
        #plt.plot(list_cumul_reward_game)
        #plt.show()
        return cumul_reward

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
