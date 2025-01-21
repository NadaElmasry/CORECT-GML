import torch
import torch.nn as nn
from torch_geometric.nn import RGCNConv, TransformerConv

class GNN(nn.Module):
    def __init__(self, g_dim, h1_dim, h2_dim, num_relations, num_modals, args):
        super(GNN, self).__init__()
        self.args = args
        self.num_modals = num_modals
        self.h1_dim = h1_dim
        self.h2_dim = h2_dim
        self.args = args

        if args.gcn_conv == "rgcn":
            print("GNN is using RGCN")
            self.rgcn_conv = RGCNConv(g_dim, h1_dim, num_relations)

        if args.use_graph_transformer:
            print("GNN is using Graph Transformer")
            in_dim = h1_dim
            self.transformer_conv = TransformerConv(in_dim, h2_dim, heads=args.graph_transformer_nheads, concat=True)
            self.bn = nn.BatchNorm1d(h2_dim * args.graph_transformer_nheads)

    def forward(self, node_features, edge_index, edge_type):

        if self.args.gcn_conv == "rgcn":
            x = self.rgcn_conv(node_features, edge_index, edge_type)

        if self.args.use_graph_transformer:
            x = self.transformer_conv(node_features, edge_index)
            x = self.bn(x.view(-1, self.h2_dim * self.args.graph_transformer_nheads)).view(x.size())
            x = nn.functional.leaky_relu(x)

        return x
