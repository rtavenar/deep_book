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

In this chapter, we will present variants of the **Gradient Descent** optimization strategy and show how they can be used to optimize neural network parameters.

Let us start with the basic Gradient Descent algorithm and its limitations.

```{prf:algorithm} Gradient Descent
:label: algo:gd

**Input:** A dataset $\mathcal{D} = (X, y)$

1. Initialize model parameters $\theta$
2. for $e = 1 .. E$

    1. for $(x_i, y_i) \in \mathcal{D}$

        1. Compute prediction $\hat{y}_i = m_\theta(x_i)$
        2. Compute gradient $\nabla_\theta \mathcal{L}_i$

    2. Compute overall gradient $\nabla_\theta \mathcal{L} = \frac{1}{n} \sum_i \nabla_\theta \mathcal{L}_i$
    3. Update parameters $\theta$ based on $\nabla_\theta \mathcal{L}$
```

The typical update rule for the parameters $\theta$ is

$$
    \theta \leftarrow \theta - \rho \nabla_\theta \mathcal{L}
$$

where $\rho$ is an important hyper-parameter of the method, called the learning rate.
Basically, gradient descent updates $\theta$ in the direction of steepest decrease of the loss $\mathcal{L}$.

As one can see in the previous algorithm, when performing gradient descent, model parameters are updated once per epoch, which means a full pass over the whole dataset is required before the update can occur.
When dealing with large datasets, this is a strong limitation, which motivates the use of stochastic variants.

## Stochastic Gradient Descent (SGD)

The idea behind the Stochastic Gradient Descent algorithm is to get cheap estimates for the quantity 

$$
    \nabla_\theta \mathcal{L}(\mathcal{D} ; m_\theta) = \frac{1}{n} \sum_{(x_i, y_i) \in \mathcal{D}} \nabla_\theta \mathcal{L}(x_i, y_i ; m_\theta)
$$

where $\mathcal{D}$ is the whole training set.
To do so, one draws subsets of data, called _minibatches_, and 

$$
    \nabla_\theta \mathcal{L}(\mathcal{B} ; m_\theta) = \frac{1}{b} \sum_{(x_i, y_i) \in \mathcal{B}} \nabla_\theta \mathcal{L}(x_i, y_i ; m_\theta)
$$
is used as an estimator for $\nabla_\theta \mathcal{L}(\mathcal{D} ; m_\theta)$.
This results in the following algorithm in which, interestingly, parameter updates occur after each minibatch, which is multiple times per epoch.

```{prf:algorithm} Stochastic Gradient Descent
:label: algo:sgd

**Input:** A dataset $\mathcal{D} = (X, y)$

1. Initialize model parameters $\theta$
2. for $e = 1 .. E$

    1. for $t = 1 .. n_\text{minibatches}$

        1. Draw minibatch $\mathcal{B}$ as a random sample of size $b$ from $\mathcal{D}$
        1. for $(x_i, y_i) \in \mathcal{B}$

            1. Compute prediction $\hat{y}_i = m_\theta(x_i)$
            2. Compute gradient $\nabla_\theta \mathcal{L}_i$

        2. Compute minibatch-level gradient $\nabla_\theta \mathcal{L}_\mathcal{B} = \frac{1}{b} \sum_i \nabla_\theta \mathcal{L}_i$
        3. Update parameters $\theta$ based on $\nabla_\theta \mathcal{L}_\mathcal{B}$
```

As a consequence, when using SGD, parameter updates are more frequent, but they are "noisy" since they are based on an minibatch estimation of the gradient instead of relying on the true gradient, as illustrated below:

```{code-cell}
:tags: [hide-input]

import numpy as np

%config InlineBackend.figure_format = 'svg'
%matplotlib inline
import matplotlib.pyplot as plt
from IPython.display import HTML

plt.ion();

import matplotlib.animation as animation
from matplotlib import rc
import scipy.optimize as optim


def grad(X, y, alpha, lambd):
    p = np.exp(-y * X.dot(alpha))
    d = - X.T.dot(p * y / (1 + p)) + lambd * alpha
    return d

def norm(x):
    return np.sqrt(np.sum(x ** 2))

def cost(X, y, alpha, lambd):
    p = np.exp(-y * X.dot(alpha))
    return np.sum(np.log(1 + p)) + .5 * lambd * norm(alpha) ** 2
    # TODO: 1/n pour pas que le SGD fasse nimp


def optim_gd(X, y, alpha_init, n_epochs, lambd, rho):
    alphas = [alpha_init]
    for _ in range(n_epochs):
        d = - grad(X, y, alphas[-1], lambd)        
        alphas.append(alphas[-1] + rho * d)

    return np.concatenate(alphas, axis=0).reshape((-1, alpha_init.shape[0]))


def optim_sgd(X, y, alpha_init, n_epochs, lambd, rho, minibatch_size):
    alphas = [alpha_init]
    for i in range(n_epochs):
        for j in range(X.shape[0] // minibatch_size):
            scaled_lambda = lambd / (X.shape[0] // minibatch_size)
            indices_minibatch = np.random.randint(X.shape[0], size=minibatch_size)
            X_minibatch = X[indices_minibatch]
            y_minibatch = y[indices_minibatch]
            d = - grad(X_minibatch, y_minibatch, alphas[-1], scaled_lambda)
              
            alphas.append(alphas[-1] + rho * d)

    return np.concatenate(alphas, axis=0).reshape((-1, alpha_init.shape[0]))


def stretch_to_range(lim, sz_range):
    middle = (lim[0] + lim[1]) / 2
    return [middle - sz_range / 2, middle + sz_range / 2]


def get_lims(*alphas_list):
    xlims = [
        min([alphas[:, 0].min() for alphas in alphas_list]) - 1,
        max([alphas[:, 0].max() for alphas in alphas_list]) + 1
    ]
    ylims = [
        min([alphas[:, 1].min() for alphas in alphas_list]) - 1,
        max([alphas[:, 1].max() for alphas in alphas_list]) + 1
    ]
    if xlims[1] - xlims[0] > ylims[1] - ylims[0]:
        ylims = stretch_to_range(ylims, xlims[1] - xlims[0])
    else:
        xlims = stretch_to_range(xlims, ylims[1] - ylims[0])
    return xlims, ylims


def gen_anim(X, y, alphas_gd, alphas_sgd, alpha_star, lambd, xlims, ylims, n_steps_per_epoch):
    global lines_alphas
    font = {'size'   : 18}
    rc('font', **font)

    n = 40
    nn = n * n
    xv, yv = np.meshgrid(np.linspace(xlims[0], xlims[1], n),
                         np.linspace(ylims[0], ylims[1], n))
    xvisu = np.concatenate((xv.ravel()[:, None], yv.ravel()[:, None]), axis=1)

    pv = np.zeros(nn)
    for i in range(nn):
        pv[i] = cost(X, y, xvisu[i], lambd)

    P = pv.reshape((n,n))
    
    fig = plt.figure(figsize=(13, 6))
    axes = [plt.subplot(1, 2, i + 1) for i in range(2)]

    lines_alphas = []
    texts = []  
    for ax, alphas, title in zip(axes, 
                                 [alphas_gd, alphas_sgd],
                                 ["Gradient Descent", "Stochastic Gradient Descent"]):
        ax.contour(xv, yv, P, alpha=0.5)
        ax.plot(alphas[0, 0], alphas[0, 1], 'ko', fillstyle='none')
        line_alphas,  = ax.plot(alphas[:1, 0], alphas[:1, 1], marker="x")
        lines_alphas.append(line_alphas)
        
        ax.plot(alpha_star[0:1], alpha_star[1:2], '+r')

        ax.set_xlabel("$w_0$")
        ax.set_ylabel("$w_1$")
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
        ax.set_title(title)
        text_epoch = ax.text(0.7 * xlims[1], 0.8 * ylims[1], s="Epoch 0")
        texts.append(text_epoch)

    def animate(i):
        global lines_alphas
        
        for line_alphas, text_epoch, alphas in zip(lines_alphas, texts, [alphas_gd, alphas_sgd]):
            line_alphas.set_xdata(alphas[:i, 0])
            line_alphas.set_ydata(alphas[:i, 1])
            
            text_epoch.set_text(f"Epoch {i // n_steps_per_epoch}")
        return lines_alphas + texts

    return animation.FuncAnimation(fig, animate, interval=500, blit=False, save_count=len(alphas_gd))


# Data

np.random.seed(0)
X = np.random.rand(20, 2) * 3 - 1.5
y = (X[:, 0] > 0.).astype(int)
y[y == 0] = -1

# Optim

lambd = .1
rho = 2e-1
alpha_init = np.array([1., -3.])
n_epochs = 10
minibatch_size = 4

res_optim = optim.minimize(fun=lambda alpha: cost(X, y, alpha, lambd),
                           x0=alpha_init, 
                           jac=lambda alpha: grad(X, y, alpha, lambd))
alpha_star = res_optim["x"]

alphas_gd = optim_gd(X, y, alpha_init, n_epochs, lambd, rho)
alphas_sgd = optim_sgd(X, y, alpha_init, n_epochs, lambd, rho, minibatch_size)

# Visualization
xlims, ylims = get_lims(alphas_gd, alphas_sgd, np.array([alpha_star]))

ani = gen_anim(X, y, 
               np.repeat(alphas_gd, 20 // minibatch_size, axis=0), alphas_sgd,
               alpha_star, lambd, xlims, ylims, 
               n_steps_per_epoch=20 // minibatch_size)
plt.close()
HTML(ani.to_jshtml())
```


Apart from implying from more frequent parameter updates, SGD has an extra benefit in terms of optimization, which is key for neural networks.
Indeed, as one can see below, contrary to what we had in the Perceptron case, the MSE loss (and the same applies for the logistic loss) is no longer convex in the model parameters as soon as the model has at least one hidden layer:

```{code-cell}
:tags: [hide-input]

def sigmoid(x):
    return 1. / (1. + np.exp(-x))

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
plt.ylabel('$\mathcal{L}$');
```

Gradient Descent is known to suffer from local optima, and such loss landscapes are a serious problem for GD.
On the other hand, Stochastic Gradient Descent is likely to benefit from noisy gradient estimations to escape local minima.

## A note on Adam

**TODO: explain formulas**

\begin{align*}
    \mathbf{m}^{(t+1)} & \propto &  \beta_1 \mathbf{m}^{(t)} + (1 - \beta_1) \nabla_\theta \mathcal{L} \\
    \mathbf{s}^{(t+1)} & \propto &  \beta_{2} \mathbf{s}^{(t)} + (1-\beta_{2}) \nabla_{\theta} \mathcal{L} \otimes \nabla_{\theta} \mathcal{L} \\
    \theta^{(t+1)} & \leftarrow & \theta^{(t)} - \rho \mathbf{m}^{(t+1)} \oslash \sqrt{\mathbf{s}^{(t+1)}+\epsilon}
\end{align*}

**TODO: illustrate SGD, SGD+momentum, Adam on a given optimization problem**

## The curse of depth

**TODO:** MLP illustration with colors and chain rule

**TODO:** A first implication: use ReLU activation functions if you have no reason to use anything else. (illustrate this?)

**TODO**: talk about feature standardization and how it eases the convergence to a good solution

## Wrapping things up in `keras`

In `keras`, loss and optimizer information are passed at compile time:


```{code-cell}
:tags: [remove-stderr]

from tensorflow.keras.layers import Dense, InputLayer
from tensorflow.keras.models import Sequential

model = Sequential([
    InputLayer(input_shape=(10, )),
    Dense(units=20, activation="relu"),
    Dense(units=3, activation="softmax")
])

model.summary()
```


```{code-cell}
:tags: [remove-stderr]

model.compile(loss="categorical_crossentropy", optimizer="adam")
```

In terms of losses:

* `"mse"` is the mean squared error loss,
* `"binary_crossentropy"` is the logistic loss for binary classification,
* `"categorical_crossentropy"` is the logistic loss for multi-class classification.

The optimizers defined in this section are available as `"sgd"` and `"adam"`.
In order to get control over optimizer hyper-parameters, one can alternatively use the following syntax:


```{code-cell}
:tags: [remove-stderr]

from tensorflow.keras.optimizers import Adam, SGD

# Not a very good idea to tune beta_1 
# and beta_2 parameters in Adam
adam_opt = Adam(learning_rate=0.001, 
                beta_1=0.9, beta_2=0.9)

# In order to use SGD with a custom learning rate:
# sgd_opt = SGD(learning_rate=0.001)

model.compile(loss="categorical_crossentropy", optimizer=adam_opt)
```
