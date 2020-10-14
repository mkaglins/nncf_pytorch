import networkx as nx
import torch
from nncf.pruning.export_helpers import Elementwise, Convolution, StopMaskForwardOps
from torch import nn as nn

from nncf.dynamic_graph.context import Scope
from nncf.pruning.utils import is_depthwise_conv, _find_next_nodes_of_types, get_sources_of_node


class LeGRBasePruner():
    def __init__(self, model, compression_algo):
        self.model = model
        self.algo = compression_algo
        self.PRUNING_QUOTA = 0.1
        self.pruned_module_info = self.algo.pruned_module_info
        self.save_weights()
        self.init_params()

    def reset(self):
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

    def init_params(self):
        """
        Resetting and initialising params for algorithm work
        :return:
        """
        self._init_next_convs_for_conv()
        self._init_chains()

        graph = self.model.get_original_graph()
        self.key_to_minfo_num = {key: self._get_minfo_num_by_scope(graph._nx_graph.nodes[key]['op_exec_context'].scope_in_model)
                                 for key in graph.get_all_node_keys() if self._get_minfo_num_by_scope(graph._nx_graph.nodes[key]['op_exec_context'].scope_in_model) is not None}

        self.conv_in_channels = {i: self.algo.pruned_module_info[i].module.in_channels for i in
                                 range(len(self.algo.pruned_module_info))}
        self.conv_out_channels = {i: self.algo.pruned_module_info[i].module.out_channels for i in
                                  range(len(self.algo.pruned_module_info))}
        self.activation_to_conv = {i: self.algo.pruned_module_info[i].module for i in
                                   range(len(self.algo.pruned_module_info))}
        self.pruning_quotas = {i: round(self.algo.pruned_module_info[i].module.out_channels * (1 - self.PRUNING_QUOTA)) for i in
                                   range(len(self.algo.pruned_module_info))}
        self.filter_ranks = {i: self.algo.filter_importance(self.algo.pruned_module_info[i].module.weight)
                             for i in range(len(self.algo.pruned_module_info))}
        self.pruning_coeffs = {i: (1, 0) for i in range(len(self.algo.pruned_module_info))}

        self._init_omap_sizes()

        # Init of params number in layers:
        graph = self.model.get_original_graph()

        # Topological sort of layers
        self.layers_top_sort = [graph._nx_graph.nodes[name] for name in nx.topological_sort(graph._nx_graph)]

        # Calculate FLOPs in all layers
        self.flops_counts = {nx_node['key']: self._get_flops_number_in_node(nx_node) for nx_node in
                             self.layers_top_sort}

        self.cost_map = {i: self._get_cost_for_conv(self._get_nx_node_by_num_in_info(i)) for i in
                               range(len(self.algo.pruned_module_info))}

        # Calculate flops in all pruned convolutions
        self.flops_in_convs = {i: self._get_flops_number_in_node(self._get_nx_node_by_num_in_info(i)) for i in
                               range(len(self.algo.pruned_module_info))}

        # Calculate FLOPs in one output filter of convolution
        self.flops_in_filter = {i: self._get_flops_number_in_filter(self._get_nx_node_by_num_in_info(i), out=True) for i in
                                range(len(self.algo.pruned_module_info))}

        # Calculate FLOPs in one input filter of convolution
        self.flops_in_input = {key: self._get_flops_number_in_filter(graph.get_nx_node_by_key(key), out=False) for key in
                               graph.get_all_node_keys() if graph.get_nx_node_by_key(key)['op_exec_context'].operator_name == 'conv2d'}
        self.full_flops = self.get_flops_number_in_model()

    def _get_cost_for_conv(self, nx_node):
        """
        Calculates FLOPs cost for conv without in/out channels.
        :param nx_node:
        :return:
        """
        graph = self.model.get_original_graph()
        nncf_node = graph._nx_node_to_nncf_node(nx_node)
        node_module = self.model.get_module_by_scope(nncf_node.op_exec_context.scope_in_model)
        if isinstance(node_module, (nn.Conv2d)):
            weight = node_module.weight
            params = weight.size(2) * weight.size(3) * self.omap_size[nx_node['key']][0] * self.omap_size[nx_node['key']][1]
        return params

    def _get_flops_number_in_node(self, nx_node):
        """
        Calculate flops number in node
        :param nx_node:
        :return:
        """
        graph = self.model.get_original_graph()
        nncf_node = graph._nx_node_to_nncf_node(nx_node)
        node_module = self.model.get_module_by_scope(nncf_node.op_exec_context.scope_in_model)
        params = 0
        if isinstance(node_module, (nn.Conv2d, nn.ConvTranspose2d)):
            w_size = node_module.weight.numel()
            params = w_size * self.omap_size[nx_node['key']][0] * self.omap_size[nx_node['key']][
                1]  # // (node_module.stride[0] ** 2)
        elif isinstance(node_module, nn.Linear):
            w_size = node_module.weight.numel()
            params = w_size
        elif isinstance(node_module, nn.BatchNorm2d):
            # TODO: FIX (multiple on H, W like Conv case)
            w_size = node_module.weight.numel()
            params = w_size
        return params

    def _get_flops_number_in_filter(self, nx_node, out):
        """
        Calculates FLOPs count in conv filter
        :param nx_node:
        :param out:
        :return:
        """
        graph = self.model.get_original_graph()
        nncf_node = graph._nx_node_to_nncf_node(nx_node)
        node_module = self.model.get_module_by_scope(nncf_node.op_exec_context.scope_in_model)
        params = 0
        if isinstance(node_module, (nn.Conv2d)):
            divider = node_module.out_channels if out else node_module.in_channels
            w_size = node_module.weight.numel() / divider
            w_size = w_size * self.omap_size[nx_node['key']][0] * self.omap_size[nx_node['key']][1]  # // (node_module.stride[0]  ** 2)
            params = w_size
        return params

    def _init_omap_sizes(self):
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
            if isinstance(node_module, (nn.Conv2d, nn.BatchNorm2d)):
                hook_list.append(node_module.register_forward_hook(get_hook(key, omap_size)))

        self.model.do_dummy_forward(force_eval=True)

        for h in hook_list:
            h.remove()
        self.pruned_modules_omap_size = pruned_modules_omap_size
        self.omap_size = omap_size

    def _get_nx_node_by_num_in_info(self, conv_num):
        graph = self.model.get_original_graph()
        conv_minfo = self.algo.pruned_module_info[conv_num]
        conv_nx_node = graph.find_node_in_nx_graph_by_scope(Scope().from_str(conv_minfo.module_name))
        return conv_nx_node

    def _get_minfo_num_by_scope(self, scope):
        for i in range(len(self.algo.pruned_module_info)):
            if scope == Scope().from_str(self.algo.pruned_module_info[i].module_name):
                return i
        return None

    def get_flops_number_in_model(self):
        return sum(self.flops_counts.values())

    def prune_layer(self, layer_counter, action, need_flops=False):
        """
        save info about coeffs for layer and return sufficient info about the next layer
        :return:
        """
        # Save coeffs
        self.pruning_coeffs[layer_counter] = torch.tensor(action[0])

        layer_counter += 1
        if layer_counter >= len(self.activation_to_conv) - 1 or not need_flops:
            flops = 0
            rest = 0
        else:
            flops = self.get_num_params_for_layer_by_top_order(layer_counter)
            rest = self.get_params_number_after_conv(layer_counter)
        return layer_counter, flops, rest

    def _find_next_convs(self, nncf_node):
        """
        Looking for next convolutions (elwises between are acceptable). All not convolutional nodes just skipped.
        :return:
        """
        next_nodes = _find_next_nodes_of_types(self.model, nncf_node,
                                               Convolution.get_all_op_aliases() + StopMaskForwardOps.get_all_op_aliases())
        conv_next_nodes = []
        for n in next_nodes:
            if n.op_exec_context.operator_name in Convolution.get_all_op_aliases():
                conv_next_nodes.append(n)
        return conv_next_nodes

    def _init_next_convs_for_conv(self):
        next_conv = {}
        graph = self.model.get_original_graph()
        for i, minfo in enumerate(self.pruned_module_info):
            conv_nx_node = self._get_nx_node_by_num_in_info(i)
            next_convs_nncf = self._find_next_convs(graph._nx_node_to_nncf_node(conv_nx_node))
            if next_convs_nncf:
                next_conv[i] = []
                for j in range(len(next_convs_nncf)):
                    nx_next_conv = graph.find_node_in_nx_graph_by_scope(next_convs_nncf[j].op_exec_context.scope_in_model)
                    next_conv[i].append(nx_next_conv['key'])
        self.next_conv = next_conv

    def get_convs_for_eltwises(self, eltwises_list):
        """
        Return all sources convs for elwises from chain
        :param eltwises_list:
        :return:
        """
        convs = []
        graph = self.model.get_original_graph()
        for el in eltwises_list:
            sources = get_sources_of_node(el, graph, Elementwise.get_all_op_aliases() +\
                                          Convolution.get_all_op_aliases() + StopMaskForwardOps.get_all_op_aliases())
            for s in sources:
                if s.op_exec_context.operator_name in Convolution.get_all_op_aliases():
                    convs.append(s)
        return convs

    def find_eltwise_chain_from_node(self, nncf_node):
        eltwise_chain = []
        next_nodes = _find_next_nodes_of_types(self.model, nncf_node,
                                               Elementwise.get_all_op_aliases() + Convolution.get_all_op_aliases() + StopMaskForwardOps.get_all_op_aliases())
        for n in next_nodes:
            if n.op_exec_context.operator_name in Elementwise.get_all_op_aliases():
                eltwise_chain.append(n)
                eltwise_chain.extend(self.find_eltwise_chain_from_node(n))
        return eltwise_chain

    def _init_chains(self):
        chains = {}
        checked = {i: False for i in range(len(self.pruned_module_info))}
        graph = self.model.get_original_graph()

        for i in range(len(self.pruned_module_info)):
            if checked[i]:
                continue

            conv_nx_node = self._get_nx_node_by_num_in_info(i)
            conv_nncf = graph._nx_node_to_nncf_node(conv_nx_node)
            cur_eltwise_chain = self.find_eltwise_chain_from_node(conv_nncf)
            cur_conv_chain = []
            if cur_eltwise_chain:
                # Usual eltwise chain case
                cur_conv_chain = self.get_convs_for_eltwises(cur_eltwise_chain)
                chains[i] = cur_conv_chain
                for n in cur_conv_chain:
                    num = self._get_minfo_num_by_scope(n.op_exec_context.scope_in_model)
                    checked[num] = True
            elif i in self.next_conv and len(self.next_conv[i]) == 1:
                # Conv + Depthwise conv case
                next_key = self.next_conv[i][0]
                nx_node = graph._nx_graph.nodes[next_key]
                module = self.model.get_module_by_scope(nx_node['op_exec_context'].scope_in_model)
                if is_depthwise_conv(module):
                    chains[i] = [conv_nncf, graph._nx_node_to_nncf_node(nx_node)]
                    num = self._get_minfo_num_by_scope(nx_node['op_exec_context'].scope_in_model)
                    if num is not None:
                        checked[num] = True
            if i not in chains:
                chains[i] = [conv_nncf]
            checked[i] = True
        self.chains = chains

    def actually_prune_all_layers(self, flops_budget):
        """
                Calculate all filter norms + scale them ->
                rank all filters and prune one by one while budget isn't met.
                :return:
                """
        filter_importances = []
        layer_indexes = []
        filter_indexes = []
        checked = {i: False for i in range(len(self.pruned_module_info))}
        graph = self.model.get_original_graph()

        for i, minfo in enumerate(self.pruned_module_info):
            if checked[i]:
                continue

            if i in self.chains:
                filter_importance = torch.zeros(minfo.module.weight.size(0)).to(minfo.module.weight.device)
                for chain_node in self.chains[i]:
                    num = self._get_minfo_num_by_scope(chain_node.op_exec_context.scope_in_model)
                    if num is not None:
                        weight = self.pruned_module_info[num].module.weight
                        filter_importance += (self.pruning_coeffs[num][0] * self.algo.filter_importance(weight) + \
                                              self.pruning_coeffs[num][1])
                        checked[num] = True
                    # TODO: uncomment
                    # filter_importance /= len(self.chains[i])
            else:
                weight = minfo.module.weight
                filter_importance = self.pruning_coeffs[i][0] * self.algo.filter_importance(weight) + \
                                    self.pruning_coeffs[i][1]
            checked[i] = True
            filter_importances.append(filter_importance)
            layer_indexes.append(i * torch.ones_like(filter_importance))
            filter_indexes.append(torch.arange(len(filter_importance)))

        importances = torch.cat(filter_importances)
        layer_indexes = torch.cat(layer_indexes)
        filter_indexes = torch.cat(filter_indexes)

        # 1. Init masks
        for i, minfo in enumerate(self.pruned_module_info):
            with torch.no_grad():
                pruning_module = minfo.operand
                pruning_module.binary_filter_pruning_mask = torch.ones(minfo.module.weight.size(0)).to(minfo.module.weight.device)

        # 2. Calculate masks
        sorted_importances = sorted(zip(importances, layer_indexes, filter_indexes), key=lambda x: x[0])
        cur_num = 0

        tmp_in_channels = dict(self.conv_in_channels)
        tmp_out_channels = dict(self.conv_out_channels)
        while cur_num < len(sorted_importances):
            layer_idx = int(sorted_importances[cur_num][1])
            filter_idx = int(sorted_importances[cur_num][2])
            if self.pruning_quotas[layer_idx] > 0:
                self.pruning_quotas[layer_idx] -= 1
            else:
                cur_num += 1
                continue

            # Prune chain
            if layer_idx in self.chains:
                chain_nodes_keys = [graph.get_node_key_by_id(nncf_node.node_id) for nncf_node in
                                    self.chains[layer_idx]]
                next_assumed = set(chain_nodes_keys)
                for node in self.chains[layer_idx]:
                    cur_idx = self._get_minfo_num_by_scope(node.op_exec_context.scope_in_model)
                    if cur_idx is None:
                        continue
                    tmp_out_channels[cur_idx] -= 1
                    pruning_module = self.pruned_module_info[cur_idx].operand
                    pruning_module.binary_filter_pruning_mask[filter_idx] = 0

                    # also need to "prune" next layers filters
                    if cur_idx in self.next_conv:
                        for next_conv_key in self.next_conv[cur_idx]:
                            if next_conv_key not in next_assumed:
                                next_conv_minfo_num = self.key_to_minfo_num[next_conv_key]
                                if not is_depthwise_conv(self.pruned_module_info[next_conv_minfo_num].module):
                                    tmp_in_channels[next_conv_minfo_num] -= 1
                            next_assumed.add(next_conv_key)
            if cur_num % 5 == 0:
                cur_flops = 0
                # Calculate current flops counts
                for key in graph.get_all_node_keys():
                    if key in self.key_to_minfo_num:
                        minfo_num = self.key_to_minfo_num[key]
                        module = self.pruned_module_info[minfo_num].module
                        cur_flops += self.cost_map[minfo_num] * tmp_in_channels[minfo_num] * tmp_out_channels[minfo_num] // module.groups
                    else:
                        # Not pruned conv node
                        cur_flops += self.flops_counts[key]
                if cur_flops < self.full_flops - flops_budget:
                    break

            cur_num += 1
        # Apply masks
        self.algo._apply_masks()
        return self.full_flops - cur_flops

    def actually_prune_all_layers_old(self, flops_budget):
        """
        Calculate all filter norms + scale them ->
        rank all filters and prune one by one while budget isn't met.
        :return:
        """
        filter_importances = []
        layer_indexes = []
        filter_indexes = []
        checked = {i: False for i in range(len(self.pruned_module_info))}
        graph = self.model.get_original_graph()

        for i, minfo in enumerate(self.pruned_module_info):
            if checked[i]:
                continue

            if i in self.chains:
                filter_importance = torch.zeros(minfo.module.weight.size(0)).to(minfo.module.weight.device)
                for chain_node in self.chains[i]:
                    num = self._get_minfo_num_by_scope(chain_node.op_exec_context.scope_in_model)
                    if num is not None:
                        weight = self.pruned_module_info[num].module.weight
                        filter_importance += (self.pruning_coeffs[num][0] * self.algo.filter_importance(weight) + \
                                self.pruning_coeffs[num][1])
                        checked[num] = True
                    # TODO: uncomment
                    # filter_importance /= len(self.chains[i])
            else:
                weight = minfo.module.weight
                filter_importance = self.pruning_coeffs[i][0] * self.algo.filter_importance(weight) + \
                                     self.pruning_coeffs[i][1]
            checked[i] = True
            filter_importances.append(filter_importance)
            layer_indexes.append(i * torch.ones_like(filter_importance))
            filter_indexes.append(torch.arange(len(filter_importance)))

        importances = torch.cat(filter_importances)
        layer_indexes = torch.cat(layer_indexes)
        filter_indexes = torch.cat(filter_indexes)

        # 1. Init masks
        for i, minfo in enumerate(self.pruned_module_info):
            pruning_module = minfo.operand
            pruning_module.binary_filter_pruning_mask = torch.ones(minfo.module.weight.size(0)).to(
                minfo.module.weight.device)

        # 2. Calculate masks
        sorted_importances = sorted(zip(importances, layer_indexes, filter_indexes), key=lambda x: x[0])
        cur_num = 0
        filters_num = 0

        remain_flops = flops_budget
        while remain_flops > 0:
            layer_idx = int(sorted_importances[cur_num][1])
            filter_idx = int(sorted_importances[cur_num][2])
            if self.pruning_quotas[layer_idx] > 0:
                self.pruning_quotas[layer_idx] -= 1
            else:
                cur_num += 1
                continue

            # ASSUME CHAINS HERE
            if layer_idx in self.chains:
                chain_nodes_keys = [graph.get_node_key_by_id(nncf_node.node_id) for nncf_node in self.chains[layer_idx]]
                next_assumed = set(chain_nodes_keys)
                for node in self.chains[layer_idx]:
                    cur_idx = self._get_minfo_num_by_scope(node.op_exec_context.scope_in_model)
                    if cur_idx is None:
                        continue
                    remain_flops -= self.flops_in_filter[cur_idx]

                    # also need to "prune" next layers filters
                    if cur_idx in self.next_conv:
                        for next_conv_key in self.next_conv[cur_idx]:
                            if next_conv_key not in next_assumed:
                                remain_flops -= self.flops_in_input[next_conv_key]
                            next_assumed.add(next_conv_key)

                    pruning_module = self.pruned_module_info[cur_idx].operand
                    pruning_module.binary_filter_pruning_mask[filter_idx] = 0
                filters_num += len(self.chains[layer_idx])
            else:
                remain_flops -= self.flops_in_filter[layer_idx]
                # also need to "prune" next layer filter
                if layer_idx in self.next_conv:
                    for next_conv_key in self.next_conv[layer_idx]:
                        remain_flops -= self.flops_in_input[next_conv_key]
                pruning_module = self.pruned_module_info[layer_idx].operand
                pruning_module.binary_filter_pruning_mask[filter_idx] = 0
                filters_num += 1
            cur_num += 1

        # Apply masks
        self.algo._apply_masks()
        return flops_budget - remain_flops
