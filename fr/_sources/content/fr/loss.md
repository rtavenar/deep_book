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
# Fonctions de coût

Nous avons maintenant présenté une première famille de modèles, qui est la famille MLP.
Afin d'entraîner ces modèles (_i.e._ d'ajuster leurs paramètres pour qu'ils s'adaptent aux données), nous devons définir une fonction de coût (aussi appelée fonction de perte, ou _loss function_) à optimiser.
Une fois cette fonction choisie, l'optimisation consistera à régler les paramètres du modèle de manière à la minimiser.

Dans cette section, nous présenterons deux fonctions de pertes standard, à savoir l'erreur quadratique moyenne (principalement utilisée pour la régression) et la fonction de perte logistique (utilisée en classification).

Dans ce qui suit, nous supposons connu un ensemble de données $\mathcal{D}$ composé de $n$ échantillons annotés $(x_i, y_i)$, et nous désignons la sortie du modèle :

$$
  \forall i, \hat{y}_i = m_\theta(x_i)
$$

où $m_\theta$ est notre modèle et $\theta$ est l'ensemble de tous ses paramètres (poids et biais).

## Erreur quadratique moyenne

L'erreur quadratique moyenne (ou _Mean Squared Error_, MSE) est la fonction de perte la plus couramment utilisée dans les contextes de régression.
Elle est définie comme suit

\begin{align*}
  \mathcal{L}(\mathcal{D} ; m_\theta) &= \frac{1}{n} \sum_i \|\hat{y}_i - y_i\|^2 \\
      &= \frac{1}{n} \sum_i \|m_{\theta}(x_i) - y_i\|^2
\end{align*}

Sa forme quadratique tend à pénaliser fortement les erreurs importantes :

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

## Perte logistique

La perte logistique est la fonction de perte la plus largement utilisée pour entraîner des réseaux neuronaux dans des contextes de classification.
Elle est définie comme suit

$$
  \mathcal{L}(\mathcal{D} ; m_\theta) = \frac{1}{n} \sum_i - \log p(\hat{y}_i = y_i ; m_\theta)
$$

où $p(\hat{y}_i = y_i ; m_\theta)$ est la probabilité prédite par le modèle $m_\theta$ pour la classe correcte $y_i$.

Sa formulation tend à favoriser les cas où le modèle prédit la classe correcte avec une probabilité proche de 1, comme on peut s'y attendre :

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

