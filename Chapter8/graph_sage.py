import sys

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import to_networkx
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import NeighborLoader


def accuracy(pred_y, y):
    return ((pred_y == y).sum() / len(y)).item()


class GraphSAGE(torch.nn.Module):
    def __init__(self, dim_in, dim_h, dim_out):
        super().__init__()
        self.sage1 = SAGEConv(dim_in, dim_h)
        self.sage2 = SAGEConv(dim_h, dim_out)

    def forward(self, x, edge_index):
        h = self.sage1(x, edge_index)
        h = torch.relu(h)
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.sage2(h, edge_index)
        return F.log_softmax(h, dim=1)

    def fit(self, train_loader, epochs):
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(),
                                    lr=0.01)

        self.train()
        for epoch in range(epochs+1):
            total_loss, val_loss, acc, val_acc = 0 ,0, 0, 0
            for batch in train_loader:
                optimizer.zero_grad()
                out = self(batch.x, batch.edge_index)
                loss = criterion(out[batch.train_mask],
                                             batch.y[batch.train_mask])
                total_loss += loss
                acc += accuracy(out[batch.train_mask].
                                            argmax(dim=1), batch.y[batch.train_mask])
                loss.backward()
                optimizer.step()
                # Validation
                val_loss += criterion(out[batch.val_mask], batch.y[batch.val_mask])
                val_acc += accuracy(out[batch.val_mask].argmax(dim=1), batch.y[batch.val_mask])
                if epoch % 20 == 0:
                    print(f'Epoch {epoch:>3} | Train Loss: {loss / len(train_loader):.3f} | Train Acc: {acc/len(train_loader) * 100:>6.2f} % | Val Loss: {val_loss/len(train_loader):.2f} | Val Acc: {val_acc / len(train_loader)*100:.2f}%')

    @torch.no_grad()
    def test(self, data):
        self.eval()
        out = self(data.x, data.edge_index)
        acc = accuracy(out.argmax(dim=1)[data.test_mask], data.y[data.test_mask])
        return acc

def pubmed():
    dataset = Planetoid(root='.', name='Pubmed')
    data = dataset[0]

    print(f'Dataset: {dataset}')

    print('-------------------')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of nodes: {data.x.shape[0]}')
    print(f'Number of features: {dataset.num_features}')
    print(f'Number of classes: {dataset.num_classes}')
    print('Graph:')
    print('------')
    print(f'Training nodes: {sum(data.train_mask).item()}')
    print(f'Evaluation nodes: {sum(data.val_mask).item()}')
    print(f'Test nodes: {sum(data.test_mask).item()}')
    print(f'Edges are directed: {data.is_directed()}')
    print(f'Graph has isolated nodes: {data.has_isolated_nodes()}')
    print(f'Graph has loops: {data.has_self_loops()}')

    train_loader = NeighborLoader(
        data,
        num_neighbors = [10, 10],
        batch_size = 16,
        input_nodes = data.train_mask,
    )

    graphsage = GraphSAGE(dataset.num_features, 64, dataset.
                          num_classes)
    print(graphsage)
    graphsage.fit(train_loader, 200)

    acc = graphsage.test(data)
    print(f'GraphSAGE test accuracy: {acc * 100:.2f}%')


if __name__ == '__main__':
    if int(sys.argv[1]) == 0:
        pubmed()
