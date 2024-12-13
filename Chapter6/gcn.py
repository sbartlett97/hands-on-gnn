import sys

import numpy as np
import torch

import pandas as pd
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch_geometric.transforms as T

from torch_geometric.datasets import Planetoid
from torch_geometric.utils import degree
from collections import Counter
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import WikipediaNetwork
from sklearn.metrics import mean_squared_error, mean_absolute_error


def accuracy(pred_y, y):
    return ((pred_y == y).sum() / len(y)).item()

class GCN(torch.nn.Module):
    """Graph Convolutional Network"""
    def __init__(self, dim_in, dim_h, dim_out):
        super().__init__()
        self.gcn1 = GCNConv(dim_in, dim_h)
        self.gcn2 = GCNConv(dim_h, dim_out)

    def forward(self, x, edge_index):
        h = self.gcn1(x, edge_index)
        h = torch.relu(h)
        h = self.gcn2(h, edge_index)
        return F.log_softmax(h, dim=1)

    def fit(self, data, epochs):
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.1,
                                    weight_decay=5e-4)

        self.train()
        for epoch in range(epochs +1):
            optimizer.zero_grad()
            out = self(data.x, data.edge_index)
            loss = criterion(out[data.train_mask], data.y[data.train_mask])
            acc = accuracy(out[data.train_mask].argmax(dim=1), data.y[data.train_mask])
            loss.backward()
            optimizer.step()

            if epoch % 20 == 0:
                val_loss = criterion(out[data.val_mask], data.y[data.val_mask])
                val_acc = accuracy(out[data.val_mask].argmax(dim=1), data.y[data.val_mask])
                print(f'Epoch {epoch:>3} | Train Loss: {loss:.3f} | Train Acc: {acc*100:>5.2f}% | Val Loss: {val_loss:.2f} | Val Acc: {val_acc*100:.2f}%')

    @torch.no_grad()
    def test(self, data):
        self.eval()
        out = self(data.x, data.edge_index)
        acc = accuracy(out.argmax(dim=1)[data.test_mask], data.y[data.test_mask])
        return acc


class RegressionGCN(torch.nn.Module):
    """Graph Convolutional Network"""
    def __init__(self, dim_in, dim_h, dim_out):
        super().__init__()
        self.gcn1 = GCNConv(dim_in, dim_h*4)
        self.gcn2 = GCNConv(dim_h*4, dim_h*2)
        self.gcn3 = GCNConv(dim_h*2, dim_h)
        self.linear = torch.nn.Linear(dim_h, dim_out)

    def forward(self, x, edge_index):
        h = self.gcn1(x, edge_index)
        h = torch.relu(h)
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.gcn2(h, edge_index)
        h = torch.relu(h)
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.gcn3(h, edge_index)
        h = torch.relu(h)
        h = self.linear(h)
        return h

    def fit(self, data, epochs):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.02,
                                    weight_decay=5e-4)

        self.train()
        for epoch in range(epochs +1):
            optimizer.zero_grad()
            out = self(data.x, data.edge_index)
            loss = F.mse_loss(out.squeeze()[data.train_mask], data.y[data.train_mask].float())
            loss.backward()
            optimizer.step()

            if epoch % 20 == 0:
                val_loss = F.mse_loss(out.squeeze()[data.val_mask], data.y[data.val_mask].float())
                print(f'Epoch {epoch:>3} | Train Loss: {loss:.5f} | Val Loss: {val_loss:.5f}')

    @torch.no_grad()
    def test(self, data):
        self.eval()
        out = self(data.x, data.edge_index)
        return F.mse_loss(out.squeeze()[data.test_mask],
                   data.y[data.test_mask].float())


def classification():
    dataset = Planetoid(root=".", name="Cora")
    data = dataset[0]

    degrees = degree(data.edge_index[0]).numpy()

    numbers = Counter(degrees)

    fig, ax = plt.subplots()
    ax.set_xlabel('Node degree')
    ax.set_ylabel('Number of nodes')
    plt.bar(numbers.keys(), numbers.values())
    gcn = GCN(dataset.num_features, 16, dataset.num_classes)
    print(gcn)
    gcn.fit(data, epochs=100)

    acc = gcn.test(data)
    print(f'GCN test accuracy: {acc*100:.2f}%')


def regression():
    dataset = WikipediaNetwork(root=".", name="chameleon",
                               transform=T.RandomNodeSplit(num_val=200, num_test=500))
    data = dataset[0]

    print(f'Dataset: {dataset}')
    print('-------------------')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of nodes: {data.x.shape[0]}')
    print(f'Number of unique features: {dataset.num_features}')
    print(f'Number of classes: {dataset.num_classes}')
    df = pd.read_csv('./Chapter6/wikipedia/chameleon/musae_chameleon_target.csv')
    values = np.log10(df['target'])
    data.y = torch.tensor(values)
    gcn = RegressionGCN(dataset.num_features, 128, 1)
    print(gcn)
    gcn.fit(data, epochs=200)
    loss = gcn.test(data)
    print(f'GCN test loss: {loss:.5f}')
    out = gcn(data.x, data.edge_index)
    y_pred = out.squeeze()[data.test_mask].detach().numpy()
    mse = mean_squared_error(data.y[data.test_mask], y_pred)
    mae = mean_absolute_error(data.y[data.test_mask], y_pred)
    print('=' * 43)
    print(f'MSE = {mse:.4f} | RMSE = {np.sqrt(mse):.4f} | MAE = {mae: .4f}')
    print('=' * 43)

if __name__=='__main__':
    if int(sys.argv[1]) == 0:
        classification()
    else:
        regression()