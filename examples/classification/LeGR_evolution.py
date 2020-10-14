import queue

import matplotlib.pyplot as plt
import torch
from examples.common.utils import print_statistics
from torch import optim

from LeGRPruner import LeGRBasePruner
from RL_training_template import AgentOptimizer
from train_test_utils import test, train_steps, train
from nncf.pruning.export_helpers import Convolution, StopMaskForwardOps, Elementwise
from nncf.pruning.utils import get_sources_of_node, is_depthwise_conv, _find_next_nodes_of_types
import numpy as np
from examples.common.example_logger import logger

SCALE_SIGMA = 1


class EvolutionOptimizer(AgentOptimizer):
    def __init__(self, kwargs):
        self.mean_loss = []
        self.filter_ranks = kwargs.get('initial_filter_ranks', {})
        self.num_layers = len(self.filter_ranks)
        self.minimum_loss = 20
        self.best_perturbation = None
        self.POPULATIONS = 64
        self.GENERATIONS = kwargs.get('GENERATIONS', {})
        self.SAMPLES = 16

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
        Predict action for the last state.
        :return: action (pertrubation)
        """
        i = self.episode
        step_size = 1 - (float(i) / (self.GENERATIONS * 1.25))
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
        Returns action for the last told state
        :return:
        """
        self.episode = episode_num
        action = self._predict_action()
        self.last_perturbation = action
        return action

    def tell(self, state, reward, end_of_episode, episode_num, info):
        """
        Getting info about episode step and save it every end of episode
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
    def __init__(self, loaders, filter_pruner, model, train_steps, pruning_max, config):
        self.prune_target = pruning_max
        self.train_loader, self.train_sampler, self.val_loader = loaders
        self.test_loader = self.val_loader

        # self.model_name = model
        self.pruner = filter_pruner
        self.steps = train_steps
        self.orig_model = model
        self.orig_model = self.orig_model
        self.config = config

    def reset(self):
        self.model = self.orig_model
        self.filter_pruner = self.pruner
        self.filter_pruner.reset()
        self.model.eval()

        self.full_flops = self.filter_pruner.get_flops_number_in_model()
        self.checked = []
        self.layer_counter = 0
        self.rest = self.full_flops  # self.filter_pruner.get_params_number_after_layer(0)
        self.last_act = (1, 0)

        return torch.zeros(1), [self.full_flops, self.rest]

    def train_steps(self, steps):
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4, nesterov=True)  #nesterov=True
        train_steps(self.train_loader, self.model, criterion, optimizer, logger, self.config, steps)

    def train(self, epochs):
        model = self.model
        optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4, nesterov=True)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [int(epochs * 0.3), int(epochs * 0.6), int(epochs * 0.8)],
                                                   gamma=0.2)
        criterion = torch.nn.CrossEntropyLoss()
        return train(model, self.config,  criterion, scheduler, optimizer, self.train_loader, self.train_sampler, self.val_loader, epochs, logger)

    def get_reward(self):
        return test(self.val_loader, self.model, torch.nn.CrossEntropyLoss(), logger, self.config.device)

    def step(self, action):
        self.last_act = action
        new_state = torch.zeros(1)
        # set pruning coeffs for every layers
        for i in range(len(action)):
            self.filter_pruner.prune_layer(i, [action[i]])

        reduced = self.filter_pruner.actually_prune_all_layers(self.full_flops*self.prune_target)
        # print_statistics(self.filter_pruner.algo.statistics(), logger)
        self.train_steps(self.steps)
        # print_statistics(self.filter_pruner.algo.statistics(), logger)

        acc, loss = self.get_reward()
        done = 1
        info = [self.full_flops, reduced]
        return new_state, (acc, loss), done, info


class LeGREvoPruner(LeGRBasePruner):
    pass


def evolution_chain_agent_train(Environment, Optimizer, env_params, agent_params, GENERATIONS, PRUNING_TARGETS):
    env = Environment(*env_params)
    env.reset()
    agent_params['initial_filter_ranks'] = env.filter_pruner.filter_ranks
    agent = Optimizer(agent_params)

    accuracy = []
    loss_list = []
    for episode in range(GENERATIONS):
        logger.info('Episode {}'.format(episode))
        state, info = env.reset()

        # Beginning of the episode
        done = 0
        reward = 0
        episode_acc = 0
        episode_loss = 0
        agent.tell(state, reward, done, episode, info)

        while not done:
            action = agent.ask(episode)
            new_state, reward, done, info = env.step(action)
            acc, loss = reward
            # LOSS AS REWARD
            agent.tell(state, loss, done, episode, info)

            state = new_state
            episode_acc += acc
        episode_loss = loss
        logger.info('Testing loss = {}'.format(loss))
        accuracy.append(episode_acc)
        loss_list.append(episode_loss)

    # # Plot EA training
    logger.info('Accuracy = {}'.format(accuracy))
    logger.info('Loss = {}'.format(loss_list))
    # plt.plot(accuracy, label='acc')
    # plt.plot(loss_list, label='loss')
    # plt.legend()
    # plt.xlabel('Epoch')
    # plt.ylabel('200 steps trained')
    # plt.savefig('/home/mkaglins/work/pruning/RL_experiments_results/mobilenet_legr_model_evo_087_400.png')

    # plt.clf()

    # For best pertrubation: prune in a couple of points and train after it
    logger.info('TRAINING BEST')
    logger.info(agent.best_perturbation)
    best_action = agent.best_perturbation

    acc_for_targets = []

    for pr in PRUNING_TARGETS:
        logger.info('')
        logger.info('Training for target pruning = {}'.format(pr))
        state, info = env.reset()
        for i in range(len(best_action)):
            env.filter_pruner.prune_layer(i, [best_action[i]])
        env.filter_pruner.actually_prune_all_layers(env.full_flops * pr)
        # print_statistics(env.filter_pruner.algo.statistics(), logger)
        epochs = 60
        acc = env.train(epochs)
        print_statistics(env.filter_pruner.algo.statistics(), logger)
        acc, loss = env.get_reward()
        logger.info('For pruning = {}, acc = {}'.format(pr, acc))
        acc_for_targets.append(acc)

    logger.info('Accuracy for target points = {}'.format(acc_for_targets))

    # plt.plot(PRUNING_TARGETS, acc_for_targets)
    # plt.xlabel('Pruning level')
    # plt.ylabel('60 epochs trained accuracy')
    # plt.title('Mobilenetv2, CIFAR100, LeGR-EA')
    # plt.savefig('/home/mkaglins/work/pruning/RL_experiments_results/mobilenet_final_acc_087_400.png')
