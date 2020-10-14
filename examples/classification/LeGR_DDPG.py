from functools import partial

import torch
import numpy as np
from torch import optim

from examples.classification.LeGRPruner import LeGRBasePruner
from examples.classification.DDPGAgent_utils import ReplayBuffer, Actor, Critic
from examples.classification.RL_training_template import AgentOptimizer
import matplotlib.pyplot as plt

from examples.classification.train_test_utils import test, train_steps
from nncf.dynamic_graph.context import Scope
from nncf.pruning.export_helpers import Convolution, Elementwise, StopMaskForwardOps
from nncf.pruning.utils import traverse_function
from examples.common.example_logger import logger

EXPLORE = 100
NUM_EPISODES = 300
SCALE_SIGMA = 1


class DDPGAgentOptimizer(AgentOptimizer):
    def __init__(self, kwargs):
        # Init DDPG params
        self.b = None
        self.filter_ranks = kwargs.get('initial_filter_ranks', {})
        self.TAU = kwargs.get('TAU', 1e-2)
        self.SIGMA = kwargs.get('SIGMA', 0.5)
        self.MINI_BATCH_SIZE = kwargs.get('MINI_BATCH_SIZE', 64)
        self.EXPLORATION_RATE = 0.3

        self.actor = Actor(11, 2, 1e-4, self.TAU)
        self.critic = Critic(11, 2, 1e-3, self.TAU)
        self.replay_buffer = ReplayBuffer(6000)
        self.episode = 0
        original_dist = self.filter_ranks.copy()
        self.original_dist_stat = {}
        for k in sorted(original_dist):
            a = original_dist[k].cpu().detach().numpy()
            self.original_dist_stat[k] = {'mean': np.mean(a), 'std': np.std(a)}

    def reset_episode(self):
        """
        Reset cureent episode info
        """
        self.states_actions = []
        self.rewards = []

    def _save_episode_info(self):
        """
        Saving information about the episode
        :return:
        """
        rewards = np.max(self.rewards) * np.ones_like(self.rewards)
        for idx, (state, action) in enumerate(self.states_actions):
            if idx != len(self.states_actions) - 1:
                t = 0
                next_state = self.states_actions[idx + 1][0]
            else:
                t = 1
                next_state = torch.zeros_like(state)
            self.replay_buffer.add(state, action, rewards[idx], t, next_state)

        if not self.b:
            self.b = np.mean(rewards)
        else:
            self.b = 0.95 * self.b + (1 - 0.95) * np.mean(rewards)

    def _train_agent_step(self):
        """
        Train RL agents
        :return:
        """
        s_batch, a_batch, r_batch, t_batch, s2_batch = \
            self.replay_buffer.sample_batch(self.MINI_BATCH_SIZE)

        target_q = self.critic.predict_target(
            s2_batch, self.actor.predict_target(s2_batch))

        y_i = torch.Tensor(
            r_batch.reshape(-1, 1) + (1 - t_batch.reshape(-1, 1)) * target_q.detach().numpy().reshape(-1,
                                                                                                      1) - self.b)

        # Update the critic given the targets
        predicted_q_value = self.critic.train_step(s_batch, torch.Tensor(a_batch).view(-1, 2), y_i)

        # Update the actor policy using the sampled gradient
        policy_loss = -self.critic.predict(s_batch, self.actor.predict(s_batch))
        policy_loss = policy_loss.mean()
        self.actor.train_step(policy_loss)

        # Update target networks
        self.actor.update_target_network()
        self.critic.update_target_network()

    def _limit_and_add_noise_to_action(self, action):
        action = action.detach().numpy()
        # print('Predicted action = {}'.format(action))
        step_size = 1 - (float(self.episode) / (NUM_EPISODES))

        # Noise
        # TODO: rewrite
        key = int(self.state[0][0] * len(self.original_dist_stat))
        need_explore = np.random.binomial(1, self.EXPLORATION_RATE)
        if self.episode > EXPLORE and need_explore:
            print('EXPLORATION')
        if self.episode < EXPLORE or need_explore:
            scale = np.exp(float(np.random.normal(0, SCALE_SIGMA * step_size)))
            shift = float(np.random.normal(0, self.original_dist_stat[key]['std']))

            action[0][0] *= scale
            action[0][1] += shift
        return action

    def _predict_action(self):
        """
        Predict action for the last state
        :return:
        """
        action = self.actor.predict(self.state)
        action = self._limit_and_add_noise_to_action(action)
        return action

    def ask(self, episode_num):
        """

        :return:
        """
        self.episode = episode_num
        action = self._predict_action()
        self.states_actions.append((self.state, action))
        return action

    def tell(self, state, reward, end_of_episode, episode_num, info):
        """
        Getting info about episode step
        :return:
        """
        # save state, reward and info
        self.state = state.view(1, -1)
        self.reward = reward
        self.info = info

        self.rewards.append(reward)
        if episode_num > EXPLORE:
            self._train_agent_step()

        if end_of_episode:
            self._save_episode_info()


class LeGR_DDPG_Env():
    def __init__(self, loaders, filter_pruner, model, steps, prune_target, config):
        self.prune_target = prune_target
        self.train_loader, self.train_sampler, self.val_loader = loaders
        self.test_loader = self.val_loader

        # self.model_name = model
        self.pruner = filter_pruner
        self.steps = steps
        self.orig_model = model
        self.orig_model = self.orig_model
        self.config = config

    def reset(self):
        self.model = self.orig_model
        self.filter_pruner = self.pruner
        self.filter_pruner.reset()
        self.model.eval()

        self.full_size = self.filter_pruner.get_flops_number_in_model()
        # Using params count instead of flops
        self.full_flops = self.full_size
        self.checked = []
        self.layer_counter = 0
        self.rest = self.full_flops # self.filter_pruner.get_params_number_after_layer(0)
        self.last_act = (1, 0)

        self.max_oc = 0
        # Looking for max in/out channels counts and H/W
        for key in self.filter_pruner.conv_out_channels:
            if self.max_oc < self.filter_pruner.conv_out_channels[key]:
                self.max_oc = self.filter_pruner.conv_out_channels[key]

        self.max_ic = 0
        for key in self.filter_pruner.conv_in_channels:
            if self.max_ic < self.filter_pruner.conv_in_channels[key]:
                self.max_ic = self.filter_pruner.conv_in_channels[key]

        allh = [self.filter_pruner.pruned_modules_omap_size[t][0] for t in range(len(self.filter_pruner.activation_to_conv))]
        allw = [self.filter_pruner.pruned_modules_omap_size[t][1] for t in range(len(self.filter_pruner.activation_to_conv))]
        self.max_fh = np.max(allh)
        self.max_fw = np.max(allw)
        self.max_stride = 0
        self.max_k = 0

        for key in self.filter_pruner.activation_to_conv:
            if self.max_stride < self.filter_pruner.activation_to_conv[key].stride[0]:
                self.max_stride = self.filter_pruner.activation_to_conv[key].stride[0]
            if self.max_k < self.filter_pruner.activation_to_conv[key].weight.size(2):
                self.max_k = self.filter_pruner.activation_to_conv[key].weight.size(2)

        # Current convolution with params
        conv = self.filter_pruner.activation_to_conv[self.layer_counter]
        h = self.filter_pruner.pruned_modules_omap_size[self.layer_counter][0]
        w = self.filter_pruner.pruned_modules_omap_size[self.layer_counter][1]

        flops = self.filter_pruner.flops_in_convs[self.layer_counter]  # Params in current layer

        state = torch.Tensor([float(self.layer_counter) / len(self.filter_pruner.activation_to_conv),
                              float(self.filter_pruner.conv_out_channels[self.layer_counter]) / self.max_oc,
                              float(self.filter_pruner.conv_in_channels[self.layer_counter]) / self.max_ic,
                              float(h) / self.max_fh,
                              float(w) / self.max_fw,
                              float(conv.stride[0]) / self.max_stride,
                              float(conv.weight.size(2)) / self.max_k,
                              float(flops) / self.full_flops,
                              float(self.rest) / self.full_flops,
                              float(self.last_act[0]),
                              float(self.last_act[1])
                              ])

        return state, [self.full_flops, self.rest, flops]

    def get_reward(self):
        return test(self.val_loader, self.model, torch.nn.CrossEntropyLoss(), logger, self.config.device)

    def train_steps(self, steps):
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=1e-2, momentum=0.9, weight_decay=4e-5, nesterov=True)
        train_steps(self.train_loader, self.model, criterion, optimizer, logger, self.config, steps)

    def step(self, action):
        self.last_act = action[0]
        if self.layer_counter >= len(self.filter_pruner.activation_to_conv) - 1:
            new_state = torch.zeros(1)
            # PRUNE LAST ONE LAYER
            self.layer_counter, self.flops, self.rest = self.filter_pruner.prune_layer(self.layer_counter, action)

            print('Pruning coeffs = {}'.format(self.filter_pruner.pruning_coeffs))
            reduced = self.filter_pruner.actually_prune_all_layers(self.full_flops*self.prune_target)
            # self.filter_pruner.algo.run_batchnorm_adaptation(self.filter_pruner.algo.config)
            self.train_steps(self.steps)

            # Get reward for training
            reward, _ = self.get_reward()
            reward = reward/10
            done = 1
            info = [self.full_flops, reduced]
        else:
            # print('LAYER NUM = {}, action = {}'.format(self.layer_counter, action))
            self.layer_counter, self.flops, self.rest = self.filter_pruner.prune_layer(self.layer_counter, action)
            conv = self.filter_pruner.activation_to_conv[self.layer_counter]
            h = self.filter_pruner.pruned_modules_omap_size[self.layer_counter][0]
            w = self.filter_pruner.pruned_modules_omap_size[self.layer_counter][1]

            new_state = torch.Tensor([float(self.layer_counter) / len(self.filter_pruner.activation_to_conv),
                                  float(self.filter_pruner.conv_out_channels[self.layer_counter]) / self.max_oc,
                                  # output channels
                                  float(self.filter_pruner.conv_in_channels[self.layer_counter]) / self.max_ic,
                                  # input channels
                                  float(h) / self.max_fh,  # H
                                  float(w) / self.max_fw,  # W
                                  float(conv.stride[0]) / self.max_stride,  # stride ?
                                  float(conv.weight.size(2)) / self.max_k,  # kernel size
                                  float(self.flops) / self.full_flops,  # FLOPS in current layer
                                  # is the total number of reduced FLOPs in previous layers
                                  float(self.rest) / self.full_flops,
                                  # Rest is the number of remaining FLOPs in the following layers
                                  float(self.last_act[0]),
                                  float(self.last_act[1])
                                      ])  # action
            reward = 0
            done = 0
            info = [self.full_flops, 0]

        return new_state, reward, done, info


class LeGRDDPGPruner(LeGRBasePruner):
    def get_num_params_for_layer_by_top_order(self, conv_num):
        graph = self.model.get_original_graph()
        conv_minfo = self.algo.pruned_module_info[conv_num]
        conv_nx_node = graph.find_node_in_nx_graph_by_scope(Scope().from_str(conv_minfo.module_name))
        conv_key = conv_nx_node['key']
        return self.flops_counts[conv_key]

    def get_params_number_after_conv(self, conv_num):
        graph = self.model.get_original_graph()
        conv_minfo = self.algo.pruned_module_info[conv_num]
        conv_nx_node = graph.find_node_in_nx_graph_by_scope(Scope().from_str(conv_minfo.module_name))
        conv_key = conv_nx_node['key']
        idx = [i for i in range(len(self.layers_top_sort)) if self.layers_top_sort[i]['key'] == conv_key]
        assert len(idx) == 1
        idx = idx[0]

        params_count = sum(self.flops_counts[layer['key']] for layer in self.layers_top_sort[idx + 1:])
        return params_count


def RL_agent_train(Environment, Optimizer, env_params, agent_params):
    env = Environment(*env_params)
    agent_params['initial_filter_ranks'] = env.pruner.filter_ranks
    agent = Optimizer(agent_params)

    rewards = []

    for episode in range(NUM_EPISODES):
        print('Episode {}'.format(episode))
        state, info = env.reset()
        agent.reset_episode()

        done = 0
        reward = 0
        episode_reward = 0
        agent.tell(state, reward, done, episode, info)

        while not done:
            action = agent.ask(episode)
            new_state, reward, done, info = env.step(action)
            agent.tell(state, reward, done, episode, info)

            state = new_state
            episode_reward += reward

        rewards.append(episode_reward)

    # plt.plot(rewards)
    # plt.xlabel('Epoch')
    # plt.ylabel('BN-adapted reward')
    # plt.savefig('mobilenetv2_legr_DDPG.png')


