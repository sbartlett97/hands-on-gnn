"""Creates a basic, directed, graph using networkx
that consists of 6 edges and 6 vertices"""

import matplotlib

import networkx as nx
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')
plt.ion()
G = nx.DiGraph()
G.add_edges_from([('A', 'B'), ('A', 'C'), ('B', 'D'),
('B', 'E'), ('C', 'F'), ('C', 'G')])

plt.figure(figsize=(10, 10))
nx.draw(
    G,
    with_labels=True,
    node_color='lightblue',
    edge_color='gray',
    node_size=2000,
    font_size=15
)
plt.show(block=True)
