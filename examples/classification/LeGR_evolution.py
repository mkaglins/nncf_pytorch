import queue
import time

import torch
import numpy as np
from torch import optim

from examples.common.utils import print_statistics

from nncf.pruning.filter_pruning.functions import calculate_binary_mask

from nncf.dynamic_graph.context import Scope

from examples.classification.RL_agent import ReplayBuffer, Critic, Actor
from examples.classification.RL_training_template import AgentOptimizer
import networkx as nx
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt

SCALE_SIGMA = 1
GENERATIONS = 200


class EvolutionOptimizer(AgentOptimizer):
    def __init__(self, kwargs):
        self.mean_loss = []
        self.filter_ranks = kwargs.get('initial_filter_ranks', {})
        self.num_layers = len(self.filter_ranks)
        self.minimum_loss = 20
        self.best_perturbation = None
        self.POPULATIONS = 64
        self.SAMPLES = 16

        SCALE_SIGMA = 1
        self.MUTATE_PERCENT = 0.1
        self.index_queue = queue.Queue(self.POPULATIONS)
        self.oldest_index = None
        self.population_loss = np.zeros(0)
        self.population_data = []

        original_dist = self.filter_ranks.copy()
        self.original_dist_stat = {}
        for k in sorted(original_dist):
            a = original_dist[k].cpu().detach().numpy()
            self.original_dist_stat[k] = {'mean': np.mean(a), 'std': np.std(a)}

    def _save_episode_info(self, reward):
        loss = reward
        if np.mean(loss) < self.minimum_loss:
            self.minimum_loss = np.mean(loss)
            self.best_perturbation = self.last_perturbation

        if self.episode < self.POPULATIONS:
            self.index_queue.put(self.episode)
            self.population_data.append(self.last_perturbation)
            self.population_loss = np.append(self.population_loss, [np.mean(loss)])
        else:
            self.index_queue.put(self.oldest_index)
            self.population_data[self.oldest_index] = self.last_perturbation
            self.population_loss[self.oldest_index] = np.mean(loss)

    def _predict_action(self):
        """
        Predict action for the last state
        :return:
        """
        i = self.episode
        step_size = 1 - (float(i) / (GENERATIONS * 1.25))
        # Perturn distribution
        perturbation = []

        if i == self.POPULATIONS - 1:
            for k in sorted(self.filter_ranks.keys()):
                perturbation.append((1, 0))
        elif i < self.POPULATIONS - 1:
            for k in sorted(self.filter_ranks.keys()):
                scale = np.exp(float(np.random.normal(0, SCALE_SIGMA)))
                shift = float(np.random.normal(0, self.original_dist_stat[k]['std']))
                perturbation.append((scale, shift))
        else:
            self.mean_loss.append(np.mean(self.population_loss))
            sampled_idx = np.random.choice(self.POPULATIONS, self.SAMPLES)
            sampled_loss = self.population_loss[sampled_idx]
            winner_idx_ = np.argmin(sampled_loss)
            winner_idx = sampled_idx[winner_idx_]
            self.oldest_index = self.index_queue.get()

            # Mutate winner
            base = self.population_data[winner_idx]
            # Perturb distribution
            mnum = int(self.MUTATE_PERCENT * len(self.filter_ranks))
            mutate_candidate = np.random.choice(len(self.filter_ranks), mnum)
            for k in sorted(self.filter_ranks.keys()):
                scale = 1
                shift = 0
                if k in mutate_candidate:
                    scale = np.exp(float(np.random.normal(0, SCALE_SIGMA * step_size)))
                    shift = float(np.random.normal(0, self.original_dist_stat[k]['std']))
                perturbation.append((scale * base[k][0], shift + base[k][1]))
        return perturbation

    def ask(self, episode_num):
        """

        :return:
        """
        self.episode = episode_num
        action = self._predict_action()
        self.last_perturbation = action
        return action

    def tell(self, state, reward, end_of_episode, episode_num, info):
        """
        Getting info about episode step
        :return:
        """
        # save state, reward and info
        self.state = state
        self.reward = reward
        self.episode = episode_num
        self.info = info

        if end_of_episode:
            self._save_episode_info(reward)


class LeGREvolutionEnv():
    def __init__(self, loaders, filter_pruner, model, max_sparsity, steps, pruning_max):
        self.prune_target = pruning_max
        self.train_loader, self.val_loader = loaders
        self.test_loader = self.val_loader

        # self.model_name = model
        self.pruner = filter_pruner
        self.max_sparsity = max_sparsity
        self.steps = steps
        self.orig_model = model
        self.orig_model = self.orig_model.cuda()

    def reset(self):
        self.model = self.orig_model
        self.filter_pruner = self.pruner
        self.filter_pruner.reset()
        self.model.eval()

        self.full_size = self.filter_pruner.get_params_number_in_model()
        # Using params count instead of flops
        self.full_flops = self.full_size
        self.checked = []
        self.layer_counter = 0
        self.rest = self.full_flops # self.filter_pruner.get_params_number_after_layer(0)
        self.last_act = (1, 0)

        return torch.zeros(1), [self.full_flops, self.rest]

    def test(self, data_loader, n_img=-1):
        self.model.eval()
        correct = 0
        total = 0
        total_len = len(data_loader)
        criterion = torch.nn.CrossEntropyLoss()
        if n_img > 0 and total_len > int(np.ceil(float(n_img) / data_loader.batch_size)):
            total_len = int(np.ceil(float(n_img) / data_loader.batch_size))
        for i, (batch, label) in enumerate(data_loader):
            if i >= total_len:
                break
            batch, label = batch.to('cuda'), label.to('cuda')
            output = self.model(batch)
            loss = criterion(output, label)
            pred = output.data.max(1)[1]
            correct += pred.eq(label).sum()
            total += label.size(0)
            if (i % 100 == 0) or (i == total_len - 1):
                print('Testing | Batch ({}/{}) | Top-1: {:.2f} ({}/{})'.format(i + 1, total_len, \
                                                                               float(correct) / total * 100, correct,
                                                                               total))
        self.model.train()
        return float(correct) / total * 100, loss.item()

    def get_reward(self):
        return self.test(self.val_loader)

    def train_steps(self, steps):
        self.model.train()
        optimizer = optim.SGD(self.model.parameters(), lr=1e-2, momentum=0.9, weight_decay=4e-5, nesterov=True)
        criterion = torch.nn.CrossEntropyLoss()

        s = 0
        avg_loss = []
        iterator = iter(self.train_loader)
        while s < steps:
            try:
                batch, label = next(iterator)
            except StopIteration:
                iterator = iter(self.train_loader)
                batch, label = next(iterator)
            batch, label = batch.to('cuda'), label.to('cuda')
            optimizer.zero_grad()
            out = self.model(batch)
            loss = criterion(out, label)
            loss.backward()
            avg_loss.append(loss.item())
            optimizer.step()
            s += 1
        print('Avg Loss: {:.3f}'.format(np.mean(avg_loss)))

    def train_epoch(self, optim, criterion):
        self.model.train()
        total = 0
        top1 = 0

        data_t = 0
        train_t = 0
        total_loss = 0
        s = time.time()
        for i, (batch, label) in enumerate(self.train_loader):
            data_t += time.time() - s
            s = time.time()
            optim.zero_grad()
            batch, label = batch.to('cuda'), label.to('cuda')
            total += batch.size(0)

            out = self.model(batch)
            loss = criterion(out, label)
            loss.backward()
            total_loss += loss.item()
            optim.step()
            train_t += time.time() - s

            if (i % 100 == 0) or (i == len(self.train_loader) - 1):
                print('Batch ({}/{}) | Loss: {:.3f} | (PerBatch) Data: {:.3f}s,  Network: {:.3f}s'.format(
                    i + 1, len(self.train_loader), total_loss / (i + 1), data_t / (i + 1), train_t / (i + 1)))
            s = time.time()

    def train(self, epochs, name=""):
        model = self.model
        optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4, nesterov=True)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [int(epochs * 0.3), int(epochs * 0.6), int(epochs * 0.8)],
                                                   gamma=0.2)
        criterion = torch.nn.CrossEntropyLoss()

        for e in range(epochs):
            print('Epoch {}...'.format(e))
            print('Train')
            self.train_epoch(optimizer, criterion)

            top1, _ = self.test(self.test_loader)
            print('Test | Top-1: {:.2f}'.format(top1))
            scheduler.step()
        top1, _ = self.test(self.test_loader)
        # torch.save(model, './ckpt/{}_final.t7'.format(name))
        return top1

    def step(self, action):
        self.last_act = action
        new_state = torch.zeros(1)
        for i in range(len(action)):
            self.filter_pruner.prune_layer(i, [action[i]])

        reduced = self.filter_pruner.actually_prune_all_layers(self.full_flops*self.prune_target)
        # self.filter_pruner.algo.run_batchnorm_adaptation(self.filter_pruner.algo.config)
        self.train_steps(200)
        # print_statistics(self.filter_pruner.algo.statistics())
        # LOSS AS REWARD
        acc, loss = self.get_reward()
        done = 1
        info = [self.full_flops, reduced]
        return new_state, (acc, loss), done, info

PRUNING_MAX = 0.8
PRUNING_TARGETS = [0.1, 0.3, 0.5, 0.7, 0.9] #[0.9, 0.87, 0.84, 0.81, 0.78]


def evolution_agent_train(Environment, Optimizer, env_params, agent_params):
    env = Environment(*env_params, PRUNING_MAX)
    env.reset()
    agent_params['initial_filter_ranks'] = env.filter_pruner.filter_ranks
    agent = Optimizer(agent_params)

    rewards = []
    loss_list = []
    for episode in range(GENERATIONS):
        print('Episode {}'.format(episode))
        state, info = env.reset()
        # agent.reset_episode()

        done = 0
        reward = 0
        episode_reward = 0
        episode_loss = 0
        agent.tell(state, reward, done, episode, info)

        while not done:
            action = agent.ask(episode)
            new_state, reward, done, info = env.step(action)
            acc, loss = reward
            agent.tell(state, loss, done, episode, info)

            state = new_state
            episode_reward += acc
        episode_loss = loss
        print('LOSS = {}'.format(loss))
        rewards.append(episode_reward)
        loss_list.append(episode_loss)

    # plt.plot(rewards, label='acc')
    # plt.plot(loss_list, label='loss')
    # plt.legend()
    # plt.xlabel('Epoch')
    # plt.ylabel('BN-adapted reward')
    # plt.savefig('mobilenet_evo_03.png')

    # For best pertrubation: prune in a couple of points and train after it
    print('TRAINING BEST')
    # print(agent.best_perturbation)
    best_action = agent.best_perturbation

    for pr in PRUNING_TARGETS:
        state, info = env.reset()
        for i in range(len(best_action)):
            env.filter_pruner.prune_layer(i, [best_action[i]])
        env.filter_pruner.actually_prune_all_layers(env.full_flops * pr)

        epochs = 60
        env.train(epochs)
        acc, loss = env.get_reward()
        print('For pruning = {}, acc = {}'.format(pr, acc))
    # for pruning_reate in PRUNING_TARGETS:
    #     # Prune model
