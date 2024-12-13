import numpy as np
np.random.seed(0)

def leaky_relu(x, alpha=0.2):
    return np.maximum(alpha*x, x)

def softmax2d(x, axis):
    e = np.exp(x - np.expand_dims(np.max(x, axis=axis), axis))
    s = np.expand_dims(np.sum(e, axis=axis), axis)
    return e/s


def main():
    A = np.array([[1, 1, 1, 1],
                  [1, 1, 0, 0],
                  [1, 0, 1, 1],
                  [1, 0, 1, 1]])

    X = np.random.uniform(-1, 1, (4,4))

    W = np.random.uniform(-1, 1, (2, 4))

    W_attn = np.random.uniform(-1, 1, (1, 4))

    connections = np.where(A > 0)

    # Linear transformation
    a = W_attn @ np.concatenate([(X @ W.T)[connections[0]], (X @ W.T)[connections[1]]], axis=1).T
    print(a)

    # Apply Leaky ReLU
    e = leaky_relu(a)
    E = np.zeros(A.shape)
    E[connections[0], connections[1]] = e[0]
    print(E)

    # Normalise attention matrix
    W_alpha = softmax2d(E, 1)
    print(W_alpha)
    # Calc attention embeddings
    H = A.T @ W_alpha @ X @ W.T
    print(H)

if __name__ == '__main__':
    main()