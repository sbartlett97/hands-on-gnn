"""Creates a basic, weighted, graph using networkx
that consists of 6 edges and 6 vertices"""

import matplotlib

import networkx as nx
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')
plt.ion()
WG = nx.Graph()
WG.add_edges_from([('A', 'B', {'weight': 10}), ('A', 'C', {'weight': 90}), ('B', 'D', {'weight': 12}),
('B', 'E', {'weight': 160}), ('C', 'F', {'weight': 30}), ('C', 'G', {'weight': 1})])
labels = nx.get_edge_attributes(WG, "weight")
plt.figure(figsize=(8, 6))
nx.draw(
    WG,
    with_labels=True,
    node_color='lightblue',
    edge_color='gray',
    node_size=2000,
    font_size=15
)
plt.show(block=True)

