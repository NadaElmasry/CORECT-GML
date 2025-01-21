from torch_geometric.utils import to_undirected, add_self_loops
import torch
import torch.nn as nn
import numpy as np


from .GNN import GNN
from utils import multi_concat, feature_packing

class GraphModel(nn.Module):
    def __init__(self, g_dim, h1_dim, h2_dim, device, args):
        super(GraphModel, self).__init__()

        self.n_modals = len(args.modalities)
        self.wp = args.wp
        self.wf = args.wf
        self.device = device
        self.edge_multi = "multi" in args.edge_type
        self.edge_temp = "temp" in args.edge_type
        self.edge_type_to_idx = self.create_edge_type_mappings(args)
        self.num_relations = len(self.edge_type_to_idx)


        self.gnn = GNN(g_dim, h1_dim, h2_dim, self.num_relations, self.n_modals, args)

        print(f"GraphModel : Edge type: {args.edge_type}")
        print(f"GraphModel : Past Windows: {args.wp}")
        print(f"GraphModel : Future Windows: {args.wf}")
    def forward(self, x, lengths):
        node_features = feature_packing(x, lengths)
        node_type, edge_index, edge_type, edge_index_lengths = self.batch_graphify(lengths)
        out_gnn = self.gnn(node_features, edge_index, edge_type)
        out_gnn = multi_concat(out_gnn, lengths, self.n_modals)
        return out_gnn

    def create_edge_type_mappings(self, args):
        edge_type_to_idx = {}
        if self.edge_temp:
            for k in range(self.n_modals):
                for j in [-1, 0, 1]:
                    edge_type_to_idx[str(j) + str(k) + str(k)] = len(edge_type_to_idx)
        else:
            for k in range(self.n_modals):
                edge_type_to_idx['0' + str(k) + str(k)] = len(edge_type_to_idx)

        if self.edge_multi:
            for j in range(self.n_modals):
                for k in range(self.n_modals):
                    if j != k:
                        edge_type_to_idx['0' + str(j) + str(k)] = len(edge_type_to_idx)

        return edge_type_to_idx

    def batch_graphify(self, lengths):
        node_type, edge_index, edge_type, edge_index_lengths = [], [], [], []
        lengths = lengths.tolist()
        total_length = sum(lengths)
        sum_length = 0

        for k in range(self.n_modals):
            node_type.extend([k] * sum(lengths))

        for j, cur_len in enumerate(lengths):
            perms = self.edge_perms(cur_len, total_length)
            edge_index_lengths.append(len(perms))

            for item in perms:
                vertices = item[0]
                neighbor = item[1]
                edge_index.append(torch.tensor([vertices + sum_length, neighbor + sum_length]))

                if vertices % total_length > neighbor % total_length:
                    temporal_type = 1
                elif vertices % total_length < neighbor % total_length:
                    temporal_type = -1
                else:
                    temporal_type = 0
                edge_type.append(self.edge_type_to_idx[str(temporal_type) +
                                                       str(node_type[vertices + sum_length]) +
                                                       str(node_type[neighbor + sum_length])])

            sum_length += cur_len

        node_type = torch.tensor(node_type).long().to(self.device)
        edge_index = torch.stack(edge_index).t().contiguous().to(self.device)
        edge_type = torch.tensor(edge_type).long().to(self.device)
        edge_index_lengths = torch.tensor(edge_index_lengths).long().to(self.device)

        return node_type, edge_index, edge_type, edge_index_lengths

    def edge_perms(self, length, total_lengths):
        perms = []
        array = np.arange(length)
        for j in range(length):
            if self.wp == -1:
                start = 0
            else:
                start = max(0, j - self.wp)
            if self.wf == -1:
                end = length
            else:
                end = min(length, j + self.wf)

            eff_array = array[start:end]

            for k in range(self.n_modals):
                node_index = j + k * total_lengths
                if self.edge_temp:
                    for item in eff_array:
                        perms.append((node_index, item + k * total_lengths))
                else:
                    perms.append((node_index, node_index))
                if self.edge_multi:
                    for l in range(self.n_modals):
                        if l != k:
                            perms.append((node_index, j + l * total_lengths))

        return perms
