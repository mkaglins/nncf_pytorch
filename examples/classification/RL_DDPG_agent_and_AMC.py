import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import networkx as nx

from examples.classification.DDPGAgent_utils import ReplayBuffer, Actor, Critic
from examples.classification.LeGRPruner import LeGRBasePruner
from examples.classification.train_test_utils import test, train, train_steps
from nncf.pruning.filter_pruning.functions import calculate_binary_mask

from nncf.pruning.utils import get_rounded_pruned_element_number, is_depthwise_conv
from nncf.dynamic_graph.context import Scope

MAX_SPARSITY = 0.9


class AMCPruner(LeGRBasePruner):
    def __init__(self, model, compression_algo, max_pruning):
        super().__init__(model, compression_algo)
        self.current_flops = self.get_flops_number_in_model()
        self.max_pruning = max_pruning
        self.pruning_levels = {i: 0 for i in range(len(self.pruned_module_info))}

    def reset(self):
        super().reset()
        self.algo_checked = {i: False for i in range(len(self.pruned_module_info))}
        self.pruning_levels = {i: 0 for i in range(len(self.pruned_module_info))}
        self.current_flops = self.get_flops_number_in_model()

    def get_num_flops_for_layer_by_top_order(self, conv_num):
        graph = self.model.get_original_graph()
        flops = 0
        next_assumed = set()
        if conv_num in self.chains:
            for chain_node in self.chains[conv_num]:
                node_id = chain_node.node_id
                node_key = graph.get_node_key_by_id(node_id)
                flops += self.flops_counts[node_key]

                cur_idx = self._get_minfo_num_by_scope(chain_node.op_exec_context.scope_in_model)
                if cur_idx:
                    if cur_idx in self.next_conv:
                        for next_conv_key in self.next_conv[cur_idx]:
                            if next_conv_key not in next_assumed:
                                flops += self.flops_counts[next_conv_key]
                            next_assumed.add(next_conv_key)
        else:
            conv_minfo = self.algo.pruned_module_info[conv_num]
            conv_nx_node = graph.find_node_in_nx_graph_by_scope(Scope().from_str(conv_minfo.module_name))
            conv_key = conv_nx_node['key']
            flops = self.flops_counts[conv_key]
        return flops

    def really_prune_model(self):
        for i, minfo in enumerate(self.algo.pruned_module_info):
            pruning_module = minfo.operand
            # 1. Calculate importance for all filters in all weights
            # 2. Calculate thresholds for every weight
            # 3. Set binary masks for filter
            filters_importance = self.algo.filter_importance(minfo.module.weight)
            num_of_remains = self.conv_out_channels[i]
            threshold = sorted(filters_importance)[-num_of_remains:]
            pruning_module.binary_filter_pruning_mask = calculate_binary_mask(filters_importance, threshold[0])

        self.algo._apply_masks()

    def get_rest_flops(self, next_layer, next_chains):
        rest = 0
        rest_max_flops = 0
        if next_layer >= len(self.pruned_module_info):
            return rest, rest_max_flops

        tmp_in_channels = self.conv_in_channels.copy()
        tmp_out_channels = self.conv_out_channels.copy()
        rest_total_flops, rest_min_flops = 0, 0
        graph = self.model.get_original_graph()
        conv_minfo = self.algo.pruned_module_info[next_layer]
        conv_nx_node = graph.find_node_in_nx_graph_by_scope(Scope().from_str(conv_minfo.module_name))
        next_layer_conv_key = conv_nx_node['key']


        # Prune next chains
        init_in = {}
        # If filter in next_chains are prune to maximum, modify the following channels
        next_assumed = set()
        for t in next_chains:
            minfo_num = self._get_minfo_num_by_scope(t.op_exec_context.scope_in_model)
            if minfo_num in self.next_conv:
                for next_conv_i in self.next_conv:
                    for next_conv_key in self.next_conv[minfo_num]:
                        if next_conv_key not in next_assumed:
                            next_conv_num = self._get_minfo_num_by_scope(
                                graph._nx_graph.nodes[next_conv_key]['op_exec_context'].scope_in_model)
                            if next_conv_num:#and not is_depthwise_conv(self.pruned_module_info[next_conv_num].module):
                                tmp_in_channels[next_conv_num] *= (1 - self.max_pruning)
                                init_in[next_conv_num] = self.conv_in_channels[next_conv_i] * (1 - self.max_pruning)
                            next_assumed.add(next_conv_key)

        next_assumed = set()
        next_layer_idx = [i for i in range(len(self.layers_top_sort)) if self.layers_top_sort[i]['key'] == next_layer_conv_key][0]
        for i in range(next_layer_idx + 1, len(self.layers_top_sort)):
            layer_key = self.layers_top_sort[i]['key']

            # если слой еще не проверен - добавляем его в rest
            nx_node = graph._nx_graph.nodes[layer_key]
            nncf_node = graph._nx_node_to_nncf_node(nx_node)
            minfo_num = self._get_minfo_num_by_scope(nx_node['op_exec_context'].scope_in_model)

            if (minfo_num is not None and not self.algo_checked[minfo_num]) or minfo_num is None:
                rest += self.flops_counts[layer_key]

                # Дальше веселее:)
                if minfo_num:
                    if nncf_node not in next_chains:
                        # режем out channels
                        tmp_out_channels[minfo_num] *= (1 - self.max_pruning)

                        # режем in channels у следующих
                        if minfo_num in self.next_conv:
                            for next_conv_key in self.next_conv[minfo_num]:
                                if next_conv_key not in next_assumed:
                                    next_conv_num = self._get_minfo_num_by_scope(graph._nx_graph.nodes[next_conv_key]['op_exec_context'].scope_in_model)
                                    if next_conv_num and not is_depthwise_conv(self.pruned_module_info[next_conv_num].module):
                                        tmp_in_channels[next_conv_num] *= (1 - self.max_pruning)
                                next_assumed.add(next_conv_key)

                        module = self.pruned_module_info[minfo_num].module
                        if minfo_num in init_in:
                            rest_total_flops += (self.cost_map[minfo_num] * init_in[minfo_num] *
                                                 self.conv_out_channels[minfo_num]) // module.groups
                        else:
                            rest_total_flops += (self.cost_map[minfo_num] * self.conv_in_channels[minfo_num] * self.conv_out_channels[minfo_num]) // module.groups
                        rest_min_flops += (self.cost_map[minfo_num] * tmp_out_channels[minfo_num] * tmp_in_channels[minfo_num]) // module.groups
        rest_max_flops = rest_total_flops - rest_min_flops
        return rest, rest_max_flops

    def prune_layer(self, layer_counter, action):
        """
        Prune layer with layer counter num with pruning level == action.
        :return:
        next_layer - idx of the next layer
        cost - то, сколько еще осталось во всей сети сейчас
        rest - сколько еще осталось в следующих после пруненой ноды слоях параметров (?)
        rest_max_flops = rest_total_flops (сколько осталось сейчас) - rest_min_flops (сколько минимально останется, то есть если попрунить по максимуму все дальше)
        """
        graph = self.model.get_original_graph()
        cost = 0

        # NX and NNCF node for curent layer_counter
        nx_layer = graph.find_node_in_nx_graph_by_scope(Scope().from_str(self.pruned_module_info[layer_counter].module_name))
        nncf_layer = graph._nx_node_to_nncf_node(nx_layer)

        # Just change number of out_channels in layer, after we need to really prune them
        old_out_channels = self.conv_out_channels[layer_counter]
        pruned_elems_num = get_rounded_pruned_element_number(self.conv_out_channels[layer_counter],
                                                             action)

        if layer_counter in self.chains:
            prune_layers_chain = self.chains[layer_counter]
        else:
            prune_layers_chain = [nncf_layer] #nncf node of layer counter

        chain_nodes_keys = [graph.get_node_key_by_id(nncf_node.node_id) for nncf_node in prune_layers_chain]
        next_assumed = set(chain_nodes_keys)
        for chain_node in prune_layers_chain:
            chain_node_num = self._get_minfo_num_by_scope(chain_node.op_exec_context.scope_in_model)
            self.algo_checked[chain_node_num] = True
            if chain_node_num is not None:
                self.pruning_levels[chain_node_num] = action
                self.conv_out_channels[chain_node_num] -= pruned_elems_num

                # also need to "prune" next layers filters
                if chain_node_num in self.next_conv:
                    for next_conv_key in self.next_conv[chain_node_num]:
                        if next_conv_key not in next_assumed:
                            next_conv_minfo_num = self.key_to_minfo_num[next_conv_key]
                            if not is_depthwise_conv(self.pruned_module_info[next_conv_minfo_num].module):
                                self.conv_in_channels[chain_node_num] -= pruned_elems_num
                        next_assumed.add(next_conv_key)

        # Calculate pruned flops with respect to new conv in/out shapes
        for key in graph.get_all_node_keys():
            if key in self.key_to_minfo_num:
                minfo_num = self.key_to_minfo_num[key]
                module = self.pruned_module_info[minfo_num].module
                cost += self.cost_map[minfo_num] * self.conv_in_channels[minfo_num] * self.conv_out_channels[
                    minfo_num] // module.groups
            else:
                # Not pruned conv node
                cost += self.flops_counts[key]

        # Calculate next layer
        next_layer = layer_counter + 1
        while next_layer in self.algo_checked and self.algo_checked[next_layer]:
            next_layer += 1

        next_chains = []
        if next_layer in self.chains:
            next_chains = self.chains[next_layer]

        # Calculate rest and rest_max
        rest, rest_max_flops = self.get_rest_flops(next_layer, next_chains)
        self.current_flops = cost
        return next_layer, cost, rest, rest_max_flops


class AMCEnv(object):
    def __init__(self, loaders, dataset, filter_pruner, model, max_sparsity, steps, logger, config):

        self.image_size = 32 if 'CIFAR' in dataset else 224
        self.train_loader, self.val_loader = loaders
        self.test_loader = self.val_loader

        if 'ImageNet' in dataset:
            num_classes = 1000
        elif 'CIFAR100' in dataset:
            num_classes = 100
        elif 'CIFAR10' in dataset:
            num_classes = 10

        # self.model_name = model
        self.pruner = filter_pruner
        self.num_cls = num_classes
        self.max_sparsity = max_sparsity
        self.steps = steps
        self.orig_model = model
        self.orig_model = self.orig_model.cuda()
        self.logger = logger
        self.config = config

    def reset(self):
        self.model = self.orig_model
        self.filter_pruner = self.pruner
        self.filter_pruner.reset()
        self.model.eval()

        self.full_size = self.filter_pruner.get_flops_number_in_model()
        # Using params count instead of flops
        self.full_flops = self.full_size  # self.filter_pruner.cur_flops
        self.checked = []
        self.layer_counter = 0
        self.reduced = 0
        self.rest = self.full_flops * self.max_sparsity
        self.last_act = 0

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

        flops = self.filter_pruner.flops_in_convs[self.layer_counter] # Params in current layer

        state = torch.Tensor([float(self.layer_counter) / len(self.filter_pruner.activation_to_conv),
                              float(self.filter_pruner.conv_out_channels[self.layer_counter]) / self.max_oc,
                              float(self.filter_pruner.conv_in_channels[self.layer_counter]) / self.max_ic,
                              float(h) / self.max_fh,
                              float(w) / self.max_fw,
                              float(conv.stride[0]) / self.max_stride,
                              float(conv.weight.size(2)) / self.max_k,
                              float(flops) / self.full_flops,
                              float(self.reduced) / self.full_flops,
                              float(self.rest) / self.full_flops,
                              self.last_act])

        return state, [self.full_flops, self.rest, self.reduced, flops]

    def train_steps(self, steps):
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=1e-2, momentum=0.9, weight_decay=4e-5, nesterov=True)
        train_steps(self.train_loader, self.model, criterion, optimizer, self.logger, self.config, steps)

    def train(self, epochs):
        model = self.model
        optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4, nesterov=True)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [int(epochs * 0.3), int(epochs * 0.6), int(epochs * 0.8)],
                                                   gamma=0.2)
        criterion = torch.nn.CrossEntropyLoss()
        return train(model, self.config, criterion, scheduler, optimizer, self.train_loader, self.train_sampler,
                     self.val_loader, epochs, self.logger)

    def test(self):
        return test(self.val_loader, self.model, torch.nn.CrossEntropyLoss(), self.logger, self.config.device)

    def step(self, action):
        self.last_act = action

        self.layer_counter, self.cost, self.rest, rest_max_flops = self.filter_pruner.prune_layer(self.layer_counter,
                                                                                                  action)
        # rest - число элементов в тех, что еще не попрунили (после текущего то есть)
        self.reduced = self.full_flops - self.cost # cost - то, сколько еще осталось во всей сети сейчас

        m_flop, m_size = 0, 0

        top1 = 0
        # TODO: rewrite this for layers (?)
        if self.layer_counter >= len(self.filter_pruner.activation_to_conv):
            # Just finish, evaluate reward (AFTER LAST LAYER CASE)
            state = torch.zeros(1)
            flops = 0
            # SOME INFO ABOUT PRUNED FILTERS AT THE END
            print('Pruning actions ', self.filter_pruner.pruning_levels)
            self.filter_pruner.really_prune_model()
            self.train_steps(self.steps)
            top1, loss = self.test()
            reward = top1
            terminal = 1
        else:
            # prune current layer
            flops = self.filter_pruner.get_num_flops_for_layer_by_top_order(self.layer_counter)

            conv = self.filter_pruner.activation_to_conv[self.layer_counter]
            h = self.filter_pruner.pruned_modules_omap_size[self.layer_counter][0]
            w = self.filter_pruner.pruned_modules_omap_size[self.layer_counter][1]

            state = torch.Tensor([float(self.layer_counter) / len(self.filter_pruner.activation_to_conv),
                                  float(self.filter_pruner.conv_out_channels[self.layer_counter]) / self.max_oc, # output channels
                                  float(self.filter_pruner.conv_in_channels[self.layer_counter]) / self.max_ic, # input channels
                                  float(h) / self.max_fh, # H
                                  float(w) / self.max_fw, # W
                                  float(conv.stride[0]) / self.max_stride,  # stride ?
                                  float(conv.weight.size(2)) / self.max_k, # kernel size
                                  float(flops) / self.full_flops, # FLOPS in current layer
                                  float(self.reduced) / self.full_flops,  # is the total number of reduced FLOPs in previous layers
                                  float(self.rest) / self.full_flops,  # Rest is the number of remaining FLOPs in the following layers
                                  self.last_act])  # action
            reward = 0
            terminal = 0

        return state, reward, top1, terminal, [self.full_flops, rest_max_flops, self.reduced, flops, m_flop, m_size]


def run_rl_agent(logger, model, compression_algo, data_loaders, config):
    while isinstance(model, nn.DataParallel):
        model = model.module

    PRUNER = AMCPruner(model, compression_algo, MAX_SPARSITY)
    prune_away = 90

    TARGET = prune_away / 100.
    TAU = 1e-2
    EPISODES = 400
    SIGMA = 0.5
    MINI_BATCH_SIZE = 64
    EXPLORE = 100
    b = None
    np.random.seed(98)

    actor = Actor(11, 1, 1e-4, TAU)
    critic = Critic(11, 1, 1e-3, TAU)

    env = AMCEnv(data_loaders, 'CIFAR100', PRUNER, model, MAX_SPARSITY, 200, logger, config)
    replay_buffer = ReplayBuffer(6000)
    best_reward = - 1000.0
    best_acc = 0
    best_actions = {}

    start = time.time()
    e_start = start
    for episode in range(EPISODES):
        logger.info('Episode {}'.format(episode))
        s, info = env.reset()
        t = False

        if episode > EXPLORE:
            SIGMA = SIGMA * 0.95

        rewards = []
        states_actions = []
        while not t:
            a = torch.clamp(actor.predict(s.view(1, -1)) + np.random.normal(0, SIGMA), 0, MAX_SPARSITY).detach().numpy()
            W_duty = TARGET * info[0] - info[1] - info[2]
            # TARGET * info[0] - need to prune in total, info[1] - rest_max_flops, info[2] - reduced
            a = np.maximum(float(W_duty) / info[3], a)
            a = np.minimum(a, MAX_SPARSITY)
            s2, r, top1, t, info = env.step(a)
            r /= 10
            states_actions.append((s, a))
            rewards.append(r)

            if episode > EXPLORE:
                s_batch, a_batch, r_batch, t_batch, s2_batch = \
                    replay_buffer.sample_batch(MINI_BATCH_SIZE)

                target_q = critic.predict_target(
                    s2_batch, actor.predict_target(s2_batch))

                y_i = torch.Tensor(
                    r_batch.reshape(-1, 1) + (1 - t_batch.reshape(-1, 1)) * target_q.detach().numpy().reshape(-1,
                                                                                                              1) - b)

                # Update the critic given the targets
                predicted_q_value = critic.train_step(s_batch, torch.Tensor(a_batch).view(-1, 1), y_i)

                # Update the actor policy using the sampled gradient
                policy_loss = -critic.predict(s_batch, actor.predict(s_batch))
                policy_loss = policy_loss.mean()
                actor.train_step(policy_loss)

                # Update target networks
                actor.update_target_network()
                critic.update_target_network()
            if t:
                if best_reward < r and info[2]/info[0] > 0.25:
                    best_reward = r
                    best_actions = env.filter_pruner.pruning_levels
                    best_acc = top1

            s = s2.clone()
        logger.info('ACTIONS = {}'.format( env.filter_pruner.pruning_levels))
        logger.info('REWARD = {}'.format(r))
        logger.info('REALLY pruned {}'.format(info[2]/info[0]))

        rewards = np.max(rewards) * np.ones_like(rewards)
        for idx, (state, action) in enumerate(states_actions):
            if idx != len(states_actions) - 1:
                t = 0
                next_state = states_actions[idx + 1][0]
            else:
                t = 1
                next_state = torch.zeros_like(state)
            replay_buffer.add(state, action, rewards[idx], t, next_state)

        if not b:
            b = np.mean(rewards)
        else:
            b = 0.95 * b + (1 - 0.95) * np.mean(rewards)

        print('Time for this episode: {:.2f}s'.format(time.time() - e_start))
        e_start = time.time()
    end = time.time()
    # print('BEST reward: {}'.format(best_reward))
    # print('BEST actions: {}'.format(best_actions))

    print('Finished. Total search time: {:.2f}h'.format((end - start) / 3600.))
    return best_reward, best_acc, best_actions
