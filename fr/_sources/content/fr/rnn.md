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

(sec:rnn)=
# Réseaux neuronaux récurrents

Les réseaux neuronaux récurrents (RNN) traitent les éléments d'une série temporelle un par un.
Typiquement, à l'instant $t$, un bloc récurrent prend en entrée :
* l'entrée courante $x_t$ et 
* un état caché $h_{t-1}$ qui a pour but de résumer les informations clés provenant de
des entrées passées $\{x_0, \dots, x_{t-1}\}$

Ce bloc retourne un état caché mis à jour $h_{t}$ :


```{tikz}
    \usetikzlibrary{arrows.meta}
    \node[draw, circle, minimum size=36pt,inner sep=0pt] (prev_rnn_cell) at  (-3, 0) {};
    \node[draw, circle, minimum size=36pt,inner sep=0pt] (rnn_cell) at  (0, 0) {};
    \node[draw, circle, minimum size=36pt,inner sep=0pt] (next_rnn_cell) at  (3, 0) {};
    \node[scale=2] (prev_prev_rnn_cell) at  (-6, 0) {\dots};
    \node[scale=2] (next_next_rnn_cell) at  (6, 0) {\dots};
    
    \node[scale=2] (h_t) at  (0, 3) {$h_t$};
    \node[scale=2] (x_t) at  (0, -3) {$x_t$};
    
    \node[scale=2] (h_tm1) at  (-3, 3) {$h_{t-1}$};
    \node[scale=2] (x_tm1) at  (-3, -3) {$x_{t-1}$};
    
    \node[scale=2] (h_tp1) at  (3, 3) {$h_{t+1}$};
    \node[scale=2] (x_tp1) at  (3, -3) {$x_{t+1}$};

    \draw[-{Stealth[length=5mm]}] (prev_rnn_cell) -- (h_tm1);
    \draw[-{Stealth[length=5mm]}] (prev_prev_rnn_cell) -- (prev_rnn_cell);
    \draw[-{Stealth[length=5mm]}] (x_tm1) -- (prev_rnn_cell);

    \draw[-{Stealth[length=5mm]}] (rnn_cell) -- (h_t);
    \draw[-{Stealth[length=5mm]}] (x_t) -- (rnn_cell);

    \draw[-{Stealth[length=5mm]}] (next_rnn_cell) -- (h_tp1);
    \draw[-{Stealth[length=5mm]}] (x_tp1) -- (next_rnn_cell);

    \draw [-{Stealth[length=5mm]}] (0, 1) .. controls (1, 1) and (1.5, 0) .. (next_rnn_cell);
    \draw [-{Stealth[length=5mm]}] (3, 1) .. controls (4, 1) and (4.5, 0) .. (next_next_rnn_cell);
    \draw [-{Stealth[length=5mm]}] (-3, 1) .. controls (-2, 1) and (-1.5, 0) .. (rnn_cell);
```


Il existe différentes couches récurrentes qui diffèrent principalement par la façon dont $h_t$ est
calculée.

```{code-cell} ipython3
:tags: [hide-cell]

%config InlineBackend.figure_format = 'svg'
%matplotlib inline
import matplotlib.pyplot as plt
from notebook_utils import prepare_notebook_graphics
prepare_notebook_graphics()
```

## Réseaux récurrents standard

La formulation originale d'une RNN est la suivante :

\begin{equation}
    \forall t, h_t = \text{tanh}(W_h h_{t-1} + W_x x_t + b)
\end{equation}

où $W_h$ est une matrice de poids associée au traitement de l'état caché précédent, $W_x$ est une autre matrice de poids associée au traitement de la
l'entrée actuelle et $b$ est un terme de biais.

On notera ici que $W_h$, $W_x$ et $b$ ne sont pas indexés par $t$, ce qui signifie que
qu'ils sont **partagés entre tous les temps**.

Une limitation importante de cette formule est qu'elle échoue à capturer les dépendances à long terme.
Pour mieux comprendre pourquoi, il faut se rappeler que les paramètres de ces réseaux sont optimisés par des  algorithmes de descente de gradient stochastique.

Pour simplifier les notations, considérons un cas simplifié dans lequel
$h_t$ et $x_t$ sont tous deux des valeurs scalaires, et regardons ce que vaut le gradient de la sortie $h_t$ par rapport à $W_h$ (qui est alors aussi un scalaire) :

\begin{equation}
    \nabla_{W_h}(h_t) = \text{tanh}^\prime(o_t) \cdot \frac{\partial o_t}{\partial W_h}
\end{equation}

où $o_t = W_h h_{t-1} + W_x x_t + b$, donc:

\begin{equation}
    \frac{\partial o_t}{\partial W_h} = h_{t-1} + W_h \cdot \frac{\partial h_{t-1}}{\partial W_h} \, .
\end{equation}

Ici, la forme de $\frac{\partial h_{t-1}}{\partial W_h}$ sera similaire à
celle de $\nabla_{W_h}(h_t)$ ci-dessus, et, au final, on obtient :

\begin{eqnarray}
    \nabla_{W_h}(h_t) &=& \text{tanh}^\prime(o_t) \cdot
        \left[
            h_{t-1} + W_h \cdot \frac{\partial h_{t-1}}{\partial W_h}
        \right] \\
        &=& \text{tanh}^\prime(o_t) \cdot
           \left[
               h_{t-1} + W_h \cdot \text{tanh}^\prime(o_{t-1}) \cdot
               \left[
                   h_{t-2} + W_h \cdot \left[ \dots \right]
               \right]
           \right] \\
          &=& h_{t-1} \text{tanh}^\prime(o_t) + h_{t-2} W_h \text{tanh}^\prime(o_t) \text{tanh}^\prime(o_{t-1}) + \dots \\
         &=& \sum_{t^\prime = 1}^{t-1} h_{t^\prime} \left[ W_h^{t-t^\prime-1} \text{tanh}^\prime(o_{t^\prime+1}) \cdot \cdots \cdot  \text{tanh}^\prime(o_{t}) \right]
\end{eqnarray}

En d'autres termes, l'influence de $h_{t^\prime}$ sera atténuée par un facteur
$W_h^{t-t^\prime-1} \text{tanh}^\prime(o_{t^\prime+1}) \cdot \cdots \cdot \text{tanh}^\prime(o_{t})$.

Rappelons maintenant à quoi ressemblent la fonction tanh et sa dérivée :

```{code-cell} ipython3
:tags: [hide-input, remove-stderr]

import torch

def tanh(x):
    return 2. / (1. + torch.exp(-2 * x)) - 1.

x = torch.linspace(-4, 4, 50, requires_grad=True)
tan_x = tanh(x)
grad_tanh_x = torch.autograd.grad(tan_x, x, grad_outputs=torch.ones_like(x), create_graph=True)[0]

plt.figure()
plt.plot(x.detach().numpy(), tan_x.detach().numpy(), label='tanh(x)')
plt.plot(x.detach().numpy(), grad_tanh_x.detach().numpy(), label='tanh\'(x)')
plt.legend()
plt.grid('on');
```

On peut voir à quel point les gradients se rapprochent rapidement de 0 pour des entrées plus grandes (en valeur absolue) que 2, et avoir plusieurs termes de ce type dans une
dérivation en chaîne fera tendre les termes correspondants vers 0.

En d'autres termes, le gradient de l'état caché au temps $t$ sera seulement
influencé par quelques uns de ses prédécesseurs $\{h_{t-1}, h_{t-2}, \dots\}$ et les
les dépendances à long terme seront ignorées lors de l'actualisation des paramètres du modèle par
descente de gradient.
Il s'agit d'une occurrence d'un phénomène plus général connu sous le nom de _vanishing gradient_.

## _Long Short Term Memory_

Les blocs _Long Short Term Memory_ (LSTM, {cite:p}`hochreiter1997long`) ont été conçus comme une alternative à aux blocs récurrents classiques.
Ils visent à atténuer l'effet de _vanishing gradient_ par l'utilisation de portes qui codent explicitement quelle partie de l'information doit (resp. ne doit pas) être utilisée.

```{admonition} Les portes dans les réseaux neuronaux
:class: tip

Dans la terminologie des réseaux de neurones, une porte $g \in [0, 1]^d$ est un vecteur utilisé pour filtrer les informations d'un vecteur caractéristique entrant $v \in \mathbb{R}^d$ de telle sorte que le résultat de l'application de la porte est : $g \odot v$.
où $\odot$ est le produit élément-par-élément.
La porte $g$ aura donc tendance à supprimer une partie des caractéristiques de $v$.
(celles qui correspondent à des valeurs très faibles de $g$).
```

Dans ces blocs, un état supplémentaire est utilisé, appelé état de la cellule $C_t$.
Cet état est calculé comme suit :

\begin{equation}
    C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t
\end{equation}

où $f_t$ est appelée _forget gate_ (elle pousse le réseau à oublier les parties inutiles de l'état passé de la cellule),
$i_t$ est l'_input gate_ et $\tilde{C}_t$ est une version actualisée de l'état de la cellule 
(qui, à son tour, peut être partiellement censurée
par l'_input gate_).

Laissons de côté pour l'instant les détails concernant le calcul de ces 3 termes et concentrons-nous plutôt sur la façon dont la formule ci-dessus est est significativement différente de la règle de mise à jour de l'état caché dans le modèle classique.
En effet, dans ce cas, si le réseau l'apprend (par l'intermédiaire de $f_t$), l'information complète de l'état précédent  de la cellule $C_{t-1}$ peut être récupérée,
ce qui permet aux gradients de se propager à rebours de l'axe du temps (et de ne plus disparaître).

Alors, le lien entre l'état de la cellule et l'état caché est :

\begin{equation}
    h_t = o_t \odot \text{tanh}(C_{t}) \, .
\end{equation}

En d'autres termes, l'état caché est la version transformée (par la fonction tanh) de l'état de la cellule,
encore censuré par une porte de sortie (_output gate_) $o_t$.

Toutes les portes utilisées dans les formules ci-dessus sont définies de manière similaire :

\begin{eqnarray}
    f_t &=& \sigma ( W_f \cdot [h_{t-1}, x_t] + b_f) \\
    i_t &=& \sigma ( W_i \cdot [h_{t-1}, x_t] + b_i) \\
    o_t &=& \sigma ( W_o \cdot [h_{t-1}, x_t] + b_o)
\end{eqnarray}

où $\sigma$ est la fonction d'activation sigmoïde
(dont les valeurs sont comprises dans $[0, 1]$) et 
$[h_{t-1}, x_t]$ la concaténation des caractéristiques $h_{t-1}$ et $x_t$.

Enfin, l'état de cellule mis à jour $\tilde{C}_t$ est calculé comme suit :

\begin{equation}
    \tilde{C}_t = \text{tanh}(W_C \cdot [h_{t-1}, x_t] + b_C) \, .
\end{equation}

Il existe dans la littérature de nombreuses variantes de ces blocs LSTM qui reposent toujours sur les mêmes principes de base.

## Gated Recurrent Unit

Une paramétrisation légèrement différente d'un bloc récurrent est utilisée dans les Gated Recurrent Units (GRU, {cite:p}`cho2014properties`).

Les GRUs reposent également sur l'utilisation de portes pour laisser (de manière adaptative) l'information circuler à travers le temps.
Une première différence significative entre les GRUs et les LSTMs est que les GRUs n'ont pas recours à l'utilisation d'un état de cellule.
Au lieu de cela, la règle de mise à jour de l'état caché est la suivante :

\begin{equation}
    h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t
\end{equation}

où $z_t$ est une porte qui équilibre (par caractéristique) la quantité d'informations
qui est conservée de l'état caché précédent avec la quantité d'informations
qui doit être mise à jour en utilisant le nouvel état caché candidat $\tilde{h}_t$,
calculé comme suit :

\begin{equation}
    \tilde{h}_t = \text{tanh}(W \cdot [r_t \odot h_{t-1}, x_t] + b) \, ,
\end{equation}

où $r_t$ est une porte supplémentaire qui peut cacher une partie de l'état caché précédent.

Les formules pour les portes $z_t$ et $r_t$ sont similaires à celles fournies pour $f_t$,
$i_t$ et $o_t$ dans le cas des LSTMs.

Une étude graphique de la capacité de ces variantes de réseaux récurrents à apprendre des dépendances à long terme est fournie
dans {cite}`madsen2019visualizing`.

## Conclusion

Dans ce chapitre et le précédent, nous avons passé en revue les architectures de réseaux de neurones qui sont utilisées pour apprendre à partir de données temporelles ou séquentielles.
En raison de contraintes de temps, nous n'avons pas abordé les modèles basés sur l'attention dans ce cours.
Nous avons présenté les modèles convolutifs qui visent à extraire des formes locales discriminantes dans les séries et les modèles récurrents qui exploitent plutôt la notion de séquence.
Concernant ces derniers, des variantes visant à faire face à l'effet de gradient évanescent ont été introduites.
Il est à noter que les modèles récurrents sont connus pour nécessiter plus de données d'entraînement que leurs homologues convolutifs.

## Références

```{bibliography}
:filter: docname in docnames
```
