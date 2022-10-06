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

(sec:loss)=
# Losses

We have now presented a first family of models, which is the MLP family.
In order to train these models (_i.e._ tune their parameters to fit the data), we need to define a loss function to be optimized.
Indeed, once this loss function is picked, optimization will consist in tuning the model parameters so as to minimize the loss.

In this section, we will present two standard losses, that are the mean squared error (that is mainly used for regression) and logistic loss (which is used in classification settings).

In the following, we assume that we are given a dataset $\mathcal{D}$ made of $n$ annotated samples $(x_i, y_i)$, and we denote the model's output:

$$
  \forall i, \hat{y}_i = m_\theta(x_i)
$$

where $m_\theta$ is our model and $\theta$ is the set of all its parameters (weights and biases).

## Mean Squared Error

The Mean Squared Error (MSE) is the most commonly used loss function in regression settings.
It is defined as:

\begin{align*}
  \mathcal{L}(\mathcal{D} ; m_\theta) &= \frac{1}{n} \sum_i \|\hat{y}_i - y_i\|^2 \\
      &= \frac{1}{n} \sum_i \|\m_{\theta}(x_i) - y_i\|^2
\end{align*}

Its quadratic formulation tends to strongly penalize large errors:

```{code-cell} ipython3
---
render:
  image:
    tex_specific_width: 60%
tags: [hide-input]
---

import numpy as np

%config InlineBackend.figure_format = 'svg'
%matplotlib inline
import matplotlib.pyplot as plt
from notebook_utils import prepare_notebook_graphics
prepare_notebook_graphics()

x = np.linspace(-4, 4, 50)

plt.plot(x, x ** 2)
plt.grid('on')
plt.xlabel("$\hat{y} - y$")
plt.ylabel("$\|\hat{y} - y\|^2$");
```

## Logistic loss

The logistic loss is the most widely used loss to train neural networks in classification settings.
It is defined as:

$$
  \mathcal{L}(\mathcal{D} ; m_\theta) = \frac{1}{n} \sum_i - \log p(\hat{y}_i = y_i ; m_\theta)
$$

where $p(\hat{y}_i = y_i ; m_\theta)$ is the probability predicted by model $m_\theta$ for the correct class $y_i$.

Its formulation tends to favor cases where the model outputs a probability of 1 for the correct class, as expected:

```{code-cell} ipython3
---
render:
  image:
    tex_specific_width: 60%
tags: [hide-input]
---

import numpy as np

%config InlineBackend.figure_format = 'svg'
%matplotlib inline
import matplotlib.pyplot as plt

plt.ion();

x = np.linspace(0.01, 1, 50)

plt.plot(x, -np.log(x))
plt.grid('on')
plt.xlabel("$p(\hat{y} = y)$")
plt.ylabel("$- \log p(\hat{y} = y)$");
```

## Loss regularizers

**Coming soon**
