import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_sort_pool, MessagePassing, GATConv, GINConv
from torch_geometric.utils import dropout_adj
import kafnets as kaf

class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(torch.nn.functional.softplus(x))

class GateConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GateConv, self).__init__(aggr='add')
        self.feature = GATConv(in_channels, out_channels, heads=4, concat=False)

        self.memory_gate = GATConv(in_channels, out_channels, heads=4, concat=False)
        self.forget_gate = GATConv(in_channels, out_channels, heads=4, concat=False)
        self.output_gate = GATConv(in_channels, out_channels, heads=4, concat=False)

        self.lin = nn.Linear(in_channels, out_channels)
        self.bn = nn.BatchNorm1d(out_channels)
        self.kaf = kaf.KAF(out_channels, D=20, kernel='gaussian')

    def reset_parameters(self):
        self.feature.reset_parameters()
        self.memory_gate.reset_parameters()
        self.forget_gate.reset_parameters()
        self.output_gate.reset_parameters()
        self.lin.reset_parameters()
        self.bn.reset_parameters()
    def forward(self, x, edge_index):

        new_feature = self.kaf(self.feature(x, edge_index))

        memory_g = torch.sigmoid(self.memory_gate(x, edge_index))
        forget_g = torch.sigmoid(self.forget_gate(x, edge_index))
        output_g = torch.sigmoid(self.output_gate(x, edge_index))


        x_transformed = self.lin(x)

        cell_state = (new_feature * memory_g) + (x_transformed * forget_g)

        out = output_g * cell_state

        return self.bn(out)

class HGRL(torch.nn.Module):
    def __init__(self, dataset, gconv=GateConv, latent_dim=[256, 128, 1], k=30,
                 dropout_n=0.4, dropout_e=0.1, force_undirected=False):
        super(HGRL, self).__init__()

        self.dropout_n = dropout_n
        self.dropout_e = dropout_e
        self.force_undirected = force_undirected

        self.conv1 = gconv(dataset.num_features, latent_dim[0])
        self.conv2 = gconv(latent_dim[0], latent_dim[1])
        self.conv3 = gconv(latent_dim[1], latent_dim[2])

        self.kaf1 = kaf.KAF(latent_dim[0], D=20, kernel='gaussian')
        self.kaf2 = kaf.KAF(latent_dim[1], D=20, kernel='gaussian')
        self.kaf3 = kaf.KAF(latent_dim[2], D=20, kernel='gaussian')

        if k < 1:
            node_nums = sorted([g.num_nodes for g in dataset])
            k = node_nums[int(math.ceil(k * len(node_nums)))-1]
            k = max(10, k)

        self.k = int(k)
        conv1d_channels = [16, 32]
        self.total_latent_dim = sum(latent_dim)
        conv1d_kws = [self.total_latent_dim, 5]
        self.conv1d_params1 = nn.Conv1d(1, conv1d_channels[0], conv1d_kws[0], conv1d_kws[0])
        self.maxpool1d = nn.MaxPool1d(2, 2)
        self.conv1d_params2 = nn.Conv1d(conv1d_channels[0], conv1d_channels[1], conv1d_kws[1], 1)

        dense_dim = int((k - 2) / 2 + 1)
        self.dense_dim = (dense_dim - conv1d_kws[1] + 1) * conv1d_channels[1]

        self.lin1 = nn.Linear(self.dense_dim, 128)
        self.lin2 = nn.Linear(128, 1)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()
        self.conv1d_params1.reset_parameters()
        self.conv1d_params2.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        edge_index, _ = dropout_adj(
            edge_index, p=self.dropout_e,
            force_undirected=self.force_undirected, num_nodes=len(x),
            training=self.training
        )

        x1 = self.kaf1(self.conv1(x, edge_index))  # Apply KAF activation after conv1
        x2 = self.kaf2(self.conv2(x1, edge_index))  # Apply KAF activation after conv2
        x3 = self.kaf3(self.conv3(x2, edge_index))  # Apply KAF activation after conv3

        X = [x1, x2, x3]
        concat_states = torch.cat(X, 1)
        x = global_sort_pool(concat_states, batch, self.k)
        x = x.unsqueeze(1)

        x = F.relu(self.conv1d_params1(x))  # Use ReLU for activation after 1D conv layer
        x = self.maxpool1d(x)
        x = F.relu(self.conv1d_params2(x))

        x = x.view(len(x), -1)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_n, training=self.training)
        x = self.lin2(x)
        return x[:, 0]
