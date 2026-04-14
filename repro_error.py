import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
import torch
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import torch.nn.functional as F

class GCN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, 16)
        self.conv2 = GCNConv(16, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

def test_plotly():
    G = nx.complete_graph(5)
    node_list = list(G.nodes())
    node_map = {node: i for i, node in enumerate(node_list)}
    edges = [[node_map[u], node_map[v]] for u, v in G.edges()]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    x = torch.randn((5, 2))
    data = Data(x=x, edge_index=edge_index)
    model = GCN(2, 2)
    model.eval()
    out = model(data)
    predictions = out.argmax(dim=1).numpy()

    pos = nx.spring_layout(G)
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(x=edge_x, y=edge_y, mode='lines')
    node_trace = go.Scatter(x=[0,1], y=[0,1], mode='markers')

    print("Testing go.Layout...")
    fig = go.Figure(data=[edge_trace, node_trace],
                 layout=go.Layout(
                    title=dict(text='Test', font=dict(size=16)),
                    showlegend=False)
                )
    print("go.Layout works!")

    print("Testing px.bar...")
    cent_fig = px.bar(x=[1,2], y=[3,4], title="Test Bar")
    print("px.bar works!")

if __name__ == "__main__":
    test_plotly()
