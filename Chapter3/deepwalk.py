import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
random.seed(0)

G = nx.erdos_renyi_graph(10, 0.3, seed=1, directed=False)

plt.figure(dpi=300)
plt.axis('off')
nx.draw_networkx(G, pos=nx.spring_layout(G, seed=0), node_size=600,
                 cmap='coolwarm', font_size=14, font_color='white'
                 )
plt.show(block=True)

def random_walk(start, length):
    """Conducts a random walk of give length from start point
    through a graph"""
    walk = [str(start)]
    for i in range(length):
        neighbors = [node for node in G.neighbors(start)]
        next_node = np.random.choice(neighbors, 1)[0]
        walk.append(str(next_node))
        start = next_node
    return walk

print(random_walk(0, 10))