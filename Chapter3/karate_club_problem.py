import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from deepwalk import random_walk
from gensim.models.word2vec import Word2Vec


random.seed(117)

G = nx.karate_club_graph()

labels = []
for node in G.nodes:
    label = G.nodes[node]['club']
    labels.append(1 if label == 'Officer' else 0)

plt.figure(figsize=(20,20), dpi=300)
plt.axis('off')
nx.draw_networkx(G, pos=nx.spring_layout(G, seed=0), node_color=labels, node_size=100,
                 cmap='coolwarm', font_size=8, font_color='white')

plt.show(block=True)

walks = []
for node in G.nodes:
    for _ in range(80):
        walks.append(random_walk(node, 10))

print(walks[0])
model = Word2Vec(walks, hs=1, sg=1, vector_size=100, window=10,
                 workers=2, seed=0)

model.train(walks, total_examples=model.corpus_count, epochs=30, report_delay=1)

print('Nodes that are the most similar to node 0:')
for similarity in model.wv.most_similar(positive=['0']):
    print(f'{similarity}')

# Similarity between two nodes
print(f"Similarity between node 0 and 4: {model.wv.similarity('0', '4')}")

nodes_wv = np.array([model.wv.get_vector(str(i)) for i in range(len(model.wv))])
labels = np.array(labels)

tsne = TSNE(n_components=2, learning_rate='auto', init='pca', random_state=0).fit_transform(nodes_wv)

plt.figure(figsize=(6, 6), dpi=300)
plt.scatter(tsne[:, 0], tsne[:, 1], s=100, c=labels, cmap="coolwarm")
plt.show(block=True)