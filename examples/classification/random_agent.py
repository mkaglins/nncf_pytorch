import os
import copy
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import networkx as nx
from nncf.pruning.filter_pruning.functions import calculate_binary_mask

from nncf.pruning.utils import get_rounded_pruned_element_number

from nncf.dynamic_graph.context import Scope

MAX_SPARSITY = 0.8
MAX_SPARSITY_2 = 0.8

class SimplePruner():
    def __init__(self, model, compression_algo, metric, num_cls):
        self.model = model
        self.algo = compression_algo
        self.metric = metric
        self.num_cls = num_cls
        self.init_params()
        self.save_weights()

    def init_params(self):
        self.conv_in_channels = {i: self.algo.pruned_module_info[i].module.weight.size(1) for i in range(len(self.algo.pruned_module_info))}
        self.conv_out_channels = {i: self.algo.pruned_module_info[i].module.weight.size(0) for i in range(len(self.algo.pruned_module_info))}
        self.activation_to_conv = {i: self.algo.pruned_module_info[i].module for i in range(len(self.algo.pruned_module_info))}
        self.pruning_levels = {i: 0 for i in range(len(self.algo.pruned_module_info))}

        omap_size = {}
        def get_hook(name):
            def compute_size_hook(module, input_, output):
                size = (output.shape[2], output.shape[3])
                omap_size[name] = size
            return compute_size_hook

        hook_list = []
        for i, minfo in enumerate(self.algo.pruned_module_info):
            hook_list.append(minfo.module.register_forward_hook(get_hook(i)))

        self.model.do_dummy_forward(force_eval=True)

        for h in hook_list:
            h.remove()
        self.omap_size = omap_size

        # Init of params number in layers:
        graph = self.model.get_original_graph()
        self.layers_top_sort = [graph._nx_graph.nodes[name] for name in nx.topological_sort(graph._nx_graph)]
        self.params_counts = {nx_node['key']: self._get_params_number_in_node(nx_node) for nx_node in self.layers_top_sort}
        self.flops_by_layer = {i: self._get_params_number_in_node(self.get_nx_by_num_in_info(i)) for i in
                               range(len(self.algo.pruned_module_info))}

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

    def get_minfo_num_by_scope(self, scope):
        for i in range(len(self.algo.pruned_module_info)):
            if scope == Scope().from_str(self.algo.pruned_module_info[i].module_name):
                return i
        return None

    def _get_params_number_in_node(self, nx_node):
        graph = self.model.get_original_graph()
        nncf_node = graph._nx_node_to_nncf_node(nx_node)
        node_module = self.model.get_module_by_scope(nncf_node.op_exec_context.scope_in_model)
        params = 0
        if isinstance(node_module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear, nn.BatchNorm2d)):
            w_size = node_module.weight.numel()
            if isinstance(node_module, (nn.Conv2d)):
                num = self.get_minfo_num_by_scope(nx_node['op_exec_context'].scope_in_model)
                if num is not None:
                    w_size = w_size * self.omap_size[num][0] * self.omap_size[num][1]
            params += w_size
        return params

    def get_nx_by_num_in_info(self, conv_num):
        graph = self.model.get_original_graph()
        conv_minfo = self.algo.pruned_module_info[conv_num]
        conv_nx_node = graph.find_node_in_nx_graph_by_scope(Scope().from_str(conv_minfo.module_name))
        return conv_nx_node

    def get_params_number_in_model(self):
        return sum(self.params_counts.values())

    def get_params_number_after_layer(self, conv_num):
        graph = self.model.get_original_graph()
        conv_minfo = self.algo.pruned_module_info[conv_num]
        conv_nx_node = graph.find_node_in_nx_graph_by_scope(Scope().from_str(conv_minfo.module_name))
        conv_key = conv_nx_node['key']
        idx = [i for i in range(len(self.layers_top_sort)) if self.layers_top_sort[i]['key'] == conv_key]
        assert len(idx) == 1
        idx = idx[0]

        params_count = sum(self.params_counts[layer['key']] for layer in self.layers_top_sort[idx + 1:])
        return params_count

    def get_num_params_for_layer(self, conv_num):
        graph = self.model.get_original_graph()
        conv_minfo = self.algo.pruned_module_info[conv_num]
        conv_nx_node = graph.find_node_in_nx_graph_by_scope(Scope().from_str(conv_minfo.module_name))
        conv_key = conv_nx_node['key']
        return self.params_counts[conv_key]

    def get_rest_max_number_of_elems(self, conv_num):
        graph = self.model.get_original_graph()
        num_params_copy = self.params_counts.copy()
        for i in range(conv_num + 1, len(self.activation_to_conv)):
            old_out_channels = self.conv_out_channels[i]
            new_out_channels = old_out_channels - get_rounded_pruned_element_number(self.conv_out_channels[i], MAX_SPARSITY_2)
            conv_minfo = self.algo.pruned_module_info[i]
            conv_nx_node = graph.find_node_in_nx_graph_by_scope(Scope().from_str(conv_minfo.module_name))
            conv_key = conv_nx_node['key']

            num_params_copy[conv_key] = round(self.params_counts[conv_key] / old_out_channels * new_out_channels)

        conv_minfo = self.algo.pruned_module_info[conv_num]
        conv_nx_node = graph.find_node_in_nx_graph_by_scope(Scope().from_str(conv_minfo.module_name))
        conv_key = conv_nx_node['key']
        idx = [i for i in range(len(self.layers_top_sort)) if self.layers_top_sort[i]['key'] == conv_key][0]

        params_count = sum(num_params_copy[layer['key']] for layer in self.layers_top_sort[idx + 1:])
        return params_count

    def reset(self):
        self.restore_weights()
        self.init_params()

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

    def compress(self, layer_counter, action, max_sparsity):
        """

        :return:
        next_layer - idx of next layer (+ 1 in my case I guess)
        cost - то, сколько еще осталось во всей сети сейчас
        rest - сколько еще осталось в следующих после пруненой ноды слоях параметров (?)
        rest_max_flops = rest_total_flops (сколько осталось сейчас) - rest_min_flops (сколько минимально останется, то есть если попрунить по максимуму все дальше)
        """
        graph = self.model.get_original_graph()
        next_layer = layer_counter + 1

        # Just change number of out_channels in layer, after we need to really prune them
        self.pruning_levels[layer_counter] = action
        old_out_channels = self.conv_out_channels[layer_counter]
        pruned_elems_num = get_rounded_pruned_element_number(self.conv_out_channels[layer_counter],
                                                                                  action)
        self.conv_out_channels[layer_counter] -= pruned_elems_num
        conv_minfo = self.algo.pruned_module_info[layer_counter]
        conv_nx_node = graph.find_node_in_nx_graph_by_scope(Scope().from_str(conv_minfo.module_name))
        conv_key = conv_nx_node['key']

        # TODO: REWRITEEE
        out_edges = list(graph._nx_graph.out_edges(conv_nx_node['key']))
        if len(out_edges) == 1:
            out_edges = list(graph._nx_graph.out_edges(out_edges[0][1]))
            if len(out_edges) == 1:
                out_edges = list(graph._nx_graph.out_edges(out_edges[0][1]))
                if len(out_edges) == 1:
                    next_node_key = out_edges[0][1]
                    next_nx_node = graph._nx_graph.nodes[next_node_key]
                    if next_nx_node['op_exec_context'].operator_name == 'conv2d':
                        num = self.get_minfo_num_by_scope(next_nx_node['op_exec_context'].scope_in_model)
                        if num is not None:
                            self.conv_in_channels[num] -= pruned_elems_num
                            self.params_counts[next_nx_node['key']] = round(self.params_counts[next_nx_node['key']]/old_out_channels * self.conv_out_channels[layer_counter])

        self.params_counts[conv_key] = round(self.params_counts[conv_key]/old_out_channels * self.conv_out_channels[layer_counter])

        cost = self.get_params_number_in_model() # calculate number of all params in network. Вопрос, как это лучше делать. Если изначально мы проходимся
        # прям по всем модулям, а тут просто для каких-то из них надо предположить, что уменьшилось число параметров.
        # Тупой вариант-хранить сколько было в начале и дальше уменьшать на то насколько попрунили

        if layer_counter >= len(self.activation_to_conv) - 1:
            rest_max_flops = 0  # rest и rest-max flops будем вместе считать
        else:
            rest_max_flops = self.get_params_number_after_layer(layer_counter + 1) - self.get_rest_max_number_of_elems(
                layer_counter + 1)
        return next_layer, cost, rest_max_flops


def measure_model(model, pruner, img_size):
    pruner.reset()
    model.eval()
    pruner.forward(torch.zeros((1, 3, img_size, img_size), device='cuda'))
    cur_flops = pruner.cur_flops
    cur_size = pruner.cur_size
    return cur_flops, cur_size


class AMCEnv(object):
    def __init__(self, loaders, dataset, filter_pruner, model, max_sparsity, metric, steps, large_input):

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
        self.metric = metric
        self.steps = steps
        self.orig_model = model
        self.orig_model = self.orig_model.cuda()

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

        flops = self.filter_pruner.flops_by_layer[self.layer_counter] # Params in current layer

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

    def train_steps(self, steps):
        self.model.train()
        optimizer = optim.SGD(self.model.parameters(), lr=4.5e-3, momentum=0.9, weight_decay=4e-5, nesterov=True)
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

    def train(self, model, epochs, name):
        self.model = model
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
        torch.save(model, './ckpt/{}_final.t7'.format(name))
        return top1

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

    def save_network(self, name):
        torch.save(self.model, '{}.t7'.format(name))

    def step(self, action):
        self.last_act = action

        self.layer_counter, self.cost, rest_max_flops = self.filter_pruner.prune_layer(self.layer_counter,
                                                                                       action,
                                                                                       self.max_sparsity)
        # rest - число элементов в тех, что еще не попрунили (после текущего то есть)
        self.reduced = self.full_flops - self.cost # cost - то, сколько еще осталось во всей сети сейчас

        top1 = 0
        if self.layer_counter >= len(self.filter_pruner.activation_to_conv):
            # Just finish, evaluate reward (AFTER LAST LAYER CASE)
            state = torch.zeros(1)
            flops = 0
            # SOME INFO ABOUT PRUNED FILTERS AT THE END
            print('Pruning actions ', self.filter_pruner.pruning_levels)
            self.filter_pruner.really_prune_model()
            self.filter_pruner.algo.run_batchnorm_adaptation()
            # self.train_steps(self.steps)
            top1, loss = self.test(self.val_loader)
            reward = loss
            terminal = 1
        else:
            reward = 0
            terminal = 0
            flops = self.filter_pruner.get_num_flops_for_layer_by_top_order(self.layer_counter)

        return reward, top1, terminal, [self.full_flops, rest_max_flops, self.reduced, flops]


def run_random_agent(logger, model, compression_algo, data_loaders, coeff, LOSS):
    while isinstance(model, nn.DataParallel):
        model = model.module

    PRUNER = SimplePruner(model, compression_algo, 'L1', 100)
    prune_away = 30

    TARGET = prune_away / 100.
    EPISODES = 400
    SIGMA = 0.8
    EXPLORE = 250
    b = None
    np.random.seed(98)

    env = AMCEnv(data_loaders, 'CIFAR100', PRUNER, model, MAX_SPARSITY, 'L1', 200, None)
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
            SIGMA = SIGMA * 0.97

        rewards = []
        states_actions = []
        while not t:
            a = np.clip(np.random.uniform(0, SIGMA), 0, MAX_SPARSITY)
            W_duty = TARGET * info[0] - info[1] - info[2]
            a = np.maximum(float(W_duty) / info[3], a)
            a = np.minimum(a, MAX_SPARSITY)
            r, top1, t, info = env.step(a)
            states_actions.append((s, a))
            if LOSS:
                r = - r - coeff * np.abs(a - TARGET)  # - np.log10(info[2]) * (100 - r)
            else:
                r = top1 - coeff * np.abs(a - TARGET)

            rewards.append(r)
            if t:
                if best_reward < r and info[2]/info[0] > 0.25:
                    best_reward = r
                    best_actions = env.filter_pruner.pruning_levels
                    best_acc = top1
        logger.info('ACTIONS = {}'.format( env.filter_pruner.pruning_levels))
        logger.info('REWARD = {}'.format(r))
        logger.info('REALLY pruned {}'.format(info[2]/info[0]))
        print('Time for this episode: {:.2f}s'.format(time.time() - e_start))
        e_start = time.time()
    end = time.time()
    print('Finished. Total search time: {:.2f}h'.format((end - start) / 3600.))
    return best_reward, best_acc, best_actions
