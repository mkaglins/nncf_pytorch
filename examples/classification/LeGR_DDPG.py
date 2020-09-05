from functools import partial

import torch
import numpy as np
from nncf.pruning.export_helpers import Convolution, Elementwise, StopMaskForwardOps

from nncf.pruning.utils import traverse_function

from examples.common.utils import print_statistics

from nncf.pruning.filter_pruning.functions import calculate_binary_mask

from nncf.dynamic_graph.context import Scope

from examples.classification.RL_agent import ReplayBuffer, Critic, Actor
from examples.classification.RL_training_template import AgentOptimizer
import networkx as nx
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt

EXPLORE = 150
NUM_EPISODES = 300
SCALE_SIGMA = 1
PRUNING_QUOTA = 0.1
class DDPGAgentOptimizer(AgentOptimizer):
    def __init__(self, kwargs):
        # Init DDPG params
        self.b = None
        self.TAU = kwargs.get('TAU', 1e-2)
        self.SIGMA = kwargs.get('SIGMA', 0.5)
        self.MINI_BATCH_SIZE = kwargs.get('MINI_BATCH_SIZE', 64)

        self.actor = Actor(11, 2, 1e-4, self.TAU)
        self.critic = Critic(11, 2, 1e-3, self.TAU)
        self.replay_buffer = ReplayBuffer(6000)
        self.episode = 0

    def reset_episode(self):
        """
        Reset cureent episode info
        """
        self.states_actions = []
        self.rewards = []

    def _save_episode_info(self):
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

    def _limit_action(self, action):
        action = action.detach().numpy()
        print('Predicted action = {}'.format(action))
        step_size = 1 - (float(self.episode) / (NUM_EPISODES))
        scale = np.exp(float(np.random.normal(0, SCALE_SIGMA * step_size)))
        shift = float(np.random.normal(0, action[0][1]))
        action[0][0] *= scale
        action[0][1] += shift
        # may be add noise
        return action

    def _predict_action(self):
        """
        Predict action for the last state
        :return:
        """
        action = self.actor.predict(self.state)
        action = self._limit_action(action)
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


class LeGRPruner():
    def __init__(self, model, compression_algo, metric):
        self.model = model
        self.algo = compression_algo
        self.metric = metric
        self.pruned_module_info = self.algo.pruned_module_info
        self.init_params()
        self.save_weights()

    def find_next_conv(self, nncf_node):
        """
        Looking for next convolution
        :return:
        """
        sources_types = Convolution.get_all_op_aliases() + Elementwise.get_all_op_aliases() + StopMaskForwardOps.get_all_op_aliases()
        graph = self.model.get_original_graph()
        visited = {node_id: False for node_id in graph.get_all_node_idxs()}
        partial_traverse_function = partial(traverse_function, nncf_graph=graph, required_types=sources_types,
                                            visited=visited)
        nncf_nodes = [nncf_node]
        if nncf_node.op_exec_context.operator_name in sources_types:
            nncf_nodes = graph.get_next_nodes(nncf_node)

        next_nodes = []
        for node in nncf_nodes:
            next_nodes.extend(graph.traverse_graph(node, partial_traverse_function))
        if len(next_nodes) == 1 and next_nodes[0].op_exec_context.operator_name in Convolution.get_all_op_aliases():
            return next_nodes[0]
        return None

    def init_next_convs_for_conv(self):
        next_conv = {}
        graph = self.model.get_original_graph()
        for i, minfo in enumerate(self.pruned_module_info):
            conv_nx_node = self.get_nx_by_num_in_info(i)
            next_conv_nncf = self.find_next_conv(graph._nx_node_to_nncf_node(conv_nx_node))
            if next_conv_nncf is not None:
                nx_next_conv = graph.find_node_in_nx_graph_by_scope(next_conv_nncf.op_exec_context.scope_in_model)
                next_conv[i] = nx_next_conv['key']
        self.next_conv = next_conv

    def init_omap_sizes(self):
        graph = self.model.get_original_graph()
        pruned_modules_omap_size = {}
        omap_size = {}

        def get_hook(name, d):
            def compute_size_hook(module, input_, output):
                size = (output.shape[2], output.shape[3])
                d[name] = size
            return compute_size_hook

        hook_list = []
        for i, minfo in enumerate(self.algo.pruned_module_info):
            hook_list.append(minfo.module.register_forward_hook(get_hook(i, pruned_modules_omap_size)))

        for key in graph.get_all_node_keys():
            node = graph.get_nx_node_by_key(key)
            nncf_node = graph._nx_node_to_nncf_node(node)
            node_module = self.model.get_module_by_scope(nncf_node.op_exec_context.scope_in_model)
            if isinstance(node_module, (nn.Conv2d)):
                hook_list.append(node_module.register_forward_hook(get_hook(key, omap_size)))

        self.model.do_dummy_forward(force_eval=True)

        for h in hook_list:
            h.remove()
        self.pruned_modules_omap_size = pruned_modules_omap_size
        self.omap_size = omap_size

    def init_params(self):
        self.init_next_convs_for_conv()
        self.conv_in_channels = {i: self.algo.pruned_module_info[i].module.weight.size(1) for i in
                                 range(len(self.algo.pruned_module_info))}
        self.conv_out_channels = {i: self.algo.pruned_module_info[i].module.weight.size(0) for i in
                                  range(len(self.algo.pruned_module_info))}
        self.activation_to_conv = {i: self.algo.pruned_module_info[i].module for i in
                                   range(len(self.algo.pruned_module_info))}
        self.pruning_quotas = {i: round(self.algo.pruned_module_info[i].module.weight.size(0) * (1 - PRUNING_QUOTA)) for i in
                                   range(len(self.algo.pruned_module_info))}
        self.filter_ranks = {i: self.algo.filter_importance(self.algo.pruned_module_info[i].module.weight)
                             for i in range(len(self.algo.pruned_module_info))}
        self.pruning_coeffs = {i: (1, 0) for i in range(len(self.algo.pruned_module_info))}

        self.init_omap_sizes()

        # Init of params number in layers:
        graph = self.model.get_original_graph()
        self.layers_top_sort = [graph._nx_graph.nodes[name] for name in nx.topological_sort(graph._nx_graph)]
        self.flops_counts = {nx_node['key']: self._get_flops_number_in_node(nx_node) for nx_node in
                             self.layers_top_sort}
        self.flops_in_convs = {i: self._get_flops_number_in_node(self.get_nx_by_num_in_info(i)) for i in
                               range(len(self.algo.pruned_module_info))}
        self.flops_in_filter = {i: self._get_flops_number_in_filter(self.get_nx_by_num_in_info(i), out=True) for i in
                               range(len(self.algo.pruned_module_info))}
        self.flops_in_input = {key: self._get_flops_number_in_filter(graph.get_nx_node_by_key(key), out=False) for key in
                               graph.get_all_node_keys() if graph.get_nx_node_by_key(key)['op_exec_context'].operator_name == 'conv2d'}
        pass

    def get_nx_by_num_in_info(self, conv_num):
        graph = self.model.get_original_graph()
        conv_minfo = self.algo.pruned_module_info[conv_num]
        conv_nx_node = graph.find_node_in_nx_graph_by_scope(Scope().from_str(conv_minfo.module_name))
        return conv_nx_node

    def get_minfo_num_by_scope(self, scope):
        for i in range(len(self.algo.pruned_module_info)):
            if scope == Scope().from_str(self.algo.pruned_module_info[i].module_name):
                return i
        return None

    def get_params_number_in_model(self):
        return sum(self.flops_counts.values())

    def _get_flops_number_in_node(self, nx_node):
        graph = self.model.get_original_graph()
        nncf_node = graph._nx_node_to_nncf_node(nx_node)
        node_module = self.model.get_module_by_scope(nncf_node.op_exec_context.scope_in_model)
        params = 0
        if isinstance(node_module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear, nn.BatchNorm2d)):
            w_size = node_module.weight.numel()
            if isinstance(node_module, (nn.Conv2d)):
                w_size = w_size * self.omap_size[nx_node['key']][0] * self.omap_size[nx_node['key']][1] // node_module.groups
            params += w_size
        return params

    def _get_flops_number_in_filter(self, nx_node, out):
        graph = self.model.get_original_graph()
        nncf_node = graph._nx_node_to_nncf_node(nx_node)
        node_module = self.model.get_module_by_scope(nncf_node.op_exec_context.scope_in_model)
        params = 0
        if isinstance(node_module, (nn.Conv2d)):
            divider = node_module.weight.size(0) if out else node_module.weight.size(1)
            w_size = node_module.weight.numel() / divider
            w_size = w_size * self.omap_size[nx_node['key']][0] * self.omap_size[nx_node['key']][1] // node_module.groups
            params += w_size
        return params

    def reset(self):
        # zeroing all params ?
        self.restore_weights()
        self.init_params()

    def save_weights(self):
        weight_list = {}
        state_dict = self.model.state_dict()
        for n, v in state_dict.items():
            weight_list[n] = v.clone()
        self.weights_clone = weight_list

    def restore_weights(self):
        state_dict = self.model.state_dict()
        for n, v in state_dict.items():
            state_dict[n].data.copy_(self.weights_clone[n].data)

    def get_num_params_for_layer(self, conv_num):
        graph = self.model.get_original_graph()
        conv_minfo = self.algo.pruned_module_info[conv_num]
        conv_nx_node = graph.find_node_in_nx_graph_by_scope(Scope().from_str(conv_minfo.module_name))
        conv_key = conv_nx_node['key']
        return self.flops_counts[conv_key]

    def get_params_number_after_layer(self, conv_num):
        graph = self.model.get_original_graph()
        conv_minfo = self.algo.pruned_module_info[conv_num]
        conv_nx_node = graph.find_node_in_nx_graph_by_scope(Scope().from_str(conv_minfo.module_name))
        conv_key = conv_nx_node['key']
        idx = [i for i in range(len(self.layers_top_sort)) if self.layers_top_sort[i]['key'] == conv_key]
        assert len(idx) == 1
        idx = idx[0]

        params_count = sum(self.flops_counts[layer['key']] for layer in self.layers_top_sort[idx + 1:])
        return params_count

    def prune_layer(self, layer_counter, action):
        """
        save info about coeffs for layer and return sufficient info about the next layer
        :return:
        """
        # Save coeffs
        self.pruning_coeffs[layer_counter] = torch.tensor(action[0])

        layer_counter += 1
        if layer_counter >= len(self.activation_to_conv) - 1:
            flops = 0
            rest = 0
        else:
            flops = self.get_num_params_for_layer(layer_counter)
            rest = self.get_params_number_after_layer(layer_counter)
        return layer_counter, flops, rest

    def actually_prune_all_layers(self, flops_budget):
        """
        Calculate all filter norms + scale them ->
        rank all filters and prune one by one while budget isn't met.
        :return:
        """
        print('COEFFS ', self.pruning_coeffs)

        all_weights = []
        filter_importances = []
        layer_indexes = []
        filter_indexes = []
        for i, minfo in enumerate(self.pruned_module_info):
            weight = minfo.module.weight
            all_weights.append(weight)
            filter_importance = self.pruning_coeffs[i][0] * self.algo.filter_importance(weight) + self.pruning_coeffs[i][1]
            filter_importances.append(filter_importance)
            layer_indexes.append(i * torch.ones_like(filter_importance))
            filter_indexes.append(torch.arange(len(filter_importance)))

        importances = torch.cat(filter_importances)
        layer_indexes = torch.cat(layer_indexes)
        filter_indexes = torch.cat(filter_indexes)

        # Calculate masks
        for i, minfo in enumerate(self.pruned_module_info):
            pruning_module = minfo.operand
            pruning_module.binary_filter_pruning_mask = torch.ones(len(filter_importances[i])).to(filter_importances[i].device)


        sorted_importances = sorted(zip(importances, layer_indexes, filter_indexes), key=lambda x: x[0])
        cur_num = 0
        remain_flops = flops_budget
        while remain_flops > 0:
            layer_idx = int(sorted_importances[cur_num][1])
            filter_idx = int(sorted_importances[cur_num][2])
            if self.pruning_quotas[layer_idx] > 0:
                self.pruning_quotas[layer_idx] -= 1
            else:
                cur_num += 1
                continue

            remain_flops -= self.flops_in_filter[layer_idx]
            # also need to "prune" next layer filter
            if layer_idx in self.next_conv:
                next_conv_key = self.next_conv[layer_idx]
                remain_flops -= self.flops_in_input[next_conv_key]

            pruning_module = self.pruned_module_info[layer_idx].operand
            pruning_module.binary_filter_pruning_mask[filter_idx] = 0

            cur_num += 1

        # Apply masks
        self.algo._apply_masks()
        return flops_budget - remain_flops


class LeGREnv():
    def __init__(self, loaders, filter_pruner, model, max_sparsity, steps, prune_target):
        self.prune_target = prune_target
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

        self.max_oc = 0
        # Looking for max in/out channels counts and H/W
        for key in self.filter_pruner.conv_out_channels:
            if self.max_oc < self.filter_pruner.conv_out_channels[key]:
                self.max_oc = self.filter_pruner.conv_out_channels[key]

        self.max_ic = 0
        for key in self.filter_pruner.conv_in_channels:
            if self.max_ic < self.filter_pruner.conv_in_channels[key]:
                self.max_ic = self.filter_pruner.conv_in_channels[key]

        allh = [self.filter_pruner.omap_size[t][0] for t in range(len(self.filter_pruner.activation_to_conv))]
        allw = [self.filter_pruner.omap_size[t][1] for t in range(len(self.filter_pruner.activation_to_conv))]
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
        h = self.filter_pruner.omap_size[self.layer_counter][0]
        w = self.filter_pruner.omap_size[self.layer_counter][1]

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

    def step(self, action):
        self.last_act = action[0]
        if self.layer_counter >= len(self.filter_pruner.activation_to_conv) - 1:
            new_state = torch.zeros(1)
            # PRUNE LAST ONE LAYER
            self.layer_counter, self.flops, self.rest = self.filter_pruner.prune_layer(self.layer_counter, action)

            reduced = self.filter_pruner.actually_prune_all_layers(self.full_flops*self.prune_target)
            self.filter_pruner.algo.run_batchnorm_adaptation(self.filter_pruner.algo.config)

            # print_statistics(self.filter_pruner.algo.statistics())
            reward, _ = self.get_reward()
            reward = reward/10
            done = 1
            info = [self.full_flops, reduced]
        else:
            print('LAYER NUM = {}, action = {}'.format(self.layer_counter, action))
            self.layer_counter, self.flops, self.rest = self.filter_pruner.prune_layer(self.layer_counter, action)
            conv = self.filter_pruner.activation_to_conv[self.layer_counter]
            h = self.filter_pruner.omap_size[self.layer_counter][0]
            w = self.filter_pruner.omap_size[self.layer_counter][1]

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


def RL_agent_train(Environment, Optimizer, env_params, agent_params):
    env = Environment(*env_params, 0.3)

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
    plt.plot(rewards)
    plt.xlabel('Epoch')
    plt.ylabel('BN-adapted reward')
    plt.savefig('resnet_18_.png')
