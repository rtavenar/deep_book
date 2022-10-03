---
jupytext:
  cell_metadata_filter: -all
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.12
    jupytext_version: 1.9.1
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

(sec:sgd)=
# Optimization

In this chapter, we will present an optimization strategy called **Gradient Descent** and its variants, and show how they can be used to optimize neural network parameters.


**Coming soon**

As one can see below, the MSE loss is no longer convex in the model parameters as soon as the model has at least one hidden layer:

```{code-cell}
:tags: [hide-cell]

import numpy as np

%config InlineBackend.figure_format = 'svg'
%matplotlib inline
import matplotlib.pyplot as plt

plt.ion();

def sigmoid(x):
    return 1. / (1. + np.exp(-x))
```

```{code-cell}
def model_forward_loss(weights, biases, X, y):
    outputs = X
    for w, b in zip(weights, biases):
        outputs = sigmoid(outputs @ w + b)
    loss = np.mean((outputs - y) ** 2)
    loss += .0001 * np.sum([(w ** 2).sum() for w in weights])
    return loss


np.random.seed(0)
w0 = np.linspace(-5, 5, 100)
X = np.random.randn(150, 6)
y = np.array([0] * 75 + [1] * 75)
weights = [
    np.random.randn(6, 20),
    np.random.randn(20, 1)
]
biases = [
    np.random.randn(1, 20),
    np.random.randn(1, 1)
]

losses = []
for wi in w0:
    weights[0][3, 9] = wi
    losses.append(model_forward_loss(weights, biases, X, y))


plt.plot(w0, losses)
plt.grid('on')
plt.xlabel('$w$')
plt.ylabel('$\mathcal{L}$')
```

<!-- **TODO: Ici, illustrer non convexité ?**

## SGD

## Variants of SGD (towards Adam)

## The curse of depth

**TODO:** A first implication: use ReLU activation functions if you have no reason to use anything else. (illustrate this?)

**TODO**: talk about feature standardization and how it eases the convergence to a good solution -->
