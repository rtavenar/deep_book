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

(sec:attention)=
# Mécanisme d'attention

Dans de nombreux contextes d'apprentissage profond (traduction automatique, résumé de texte, traitement de séquences) les modèles doivent manipuler des entrées de taille variable et se concentrer sur certaines parties plus que d'autres.

Le mécanisme d'**attention** permet justement de donner plus de poids à certains éléments d'une séquence lors du calcul d'une sortie, en fonction de leur **pertinence**.

## Motivation

Considérons la phrase suivante :

> _"An apple that had been on the tree in the garden for weeks had finally been picked up."_  

qui en français pourrait se traduire par :

> _"Une pomme qui était sur l'arbre du jardin depuis des semaines avait finalement été ramassée."_

Ici, pour bien orthographier le mot _ramassée_, il faut avoir conscience qu'il fait référence au nom _une pomme_ qui est féminin.

Pour qu'un modèle de traduction automatique soit capable d'orthographier correctement ce mot, il faut donc qu'il soit capable de modéliser des **dépendances à longue portée** entre les mots.  
Or, les architectures **récurrentes** ou **convolutives** classiques ont du mal à gérer efficacement ces dépendances à cause :
- du **goulot d’étranglement** (bottleneck) dans les représentations,
- de la difficulté à mémoriser des informations éloignées.

L'attention répond à cette limite en permettant au modèle de **se focaliser dynamiquement** sur certaines entrées au moment de produire une sortie.

## Principe général

Au lieu de résumer l’entrée par un seul vecteur fixe, comme dans les encodeurs récurrents classiques, l'attention génère une sortie en **pondérant les différentes parties de l'entrée** selon leur pertinence.

Pour chaque élément de la sortie, le modèle effectue une **agrégation pondérée** des éléments d’entrée, où les poids reflètent leur **importance**.

## Métaphore : Queries, Keys, Values

L’attention peut être interprétée via la métaphore suivante :

- **Query (Q)** : ce que l'on cherche
- **Key (K)** : ce que l'on a comme référence
- **Value (V)** : ce que l'on extrait

On peut rapprocher ce mécanisme de ce qui se passe lorqu'on manipule un dictionnaire Python :
dans un dictionnaire, on cherche une clé exacte pour obtenir la valeur associée. Ici, la requête joue le rôle de la clé recherchée, mais au lieu d’une correspondance exacte, on compare la requête à toutes les clés disponibles (qui sont des vecteurs numériques) en mesurant leur similarité (typiquement via un produit scalaire).

Plutôt que de récupérer la valeur d’une seule clé, on effectue une **moyenne pondérée** des valeurs associées aux clés les plus similaires à la requête. Les poids de cette moyenne sont justement les similarités calculées entre la requête et chaque clé.

## Formulation mathématique

Soient deux séquences de vecteurs d’entrée $X = [x_1, \dots, x_n]$ et $Y = [y_1, \dots, y_m]$.
L'attention consiste à projeter $X$ en requêtes $Q$ et $Y$ en clés $K$ et valeurs $V$ :

\begin{align*}
Q &= XW^Q \\
K &= YW^K \\
V &= YW^V
\end{align*}

où $W^Q, W^K, W^V$ sont des matrices de poids apprises.

L’attention est alors définie par :

\begin{align*}
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V
\end{align*}

où $d_k$ est la dimension des vecteurs clés (utilisé pour stabiliser l'entraînement).

```{code-cell} ipython3
import torch
import numpy as np
import torch.nn.functional as F

torch.manual_seed(0)

Q = torch.randn(1, 4, 8)  # batch, longueur, dim
K = torch.randn(1, 6, 8)  # les clés ne sont pas forcément de la même longueur
V = torch.randn(1, 6, 10) # la longueur des valeurs est celle des clés, leur dim peut être autre

scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(8)
weights = F.softmax(scores, dim=-1)
output = torch.matmul(weights, V)

output.shape  # (1, 4, 10)
```

## Auto-attention (ou self-attention)

Dans certains cas, comme en traitement de séquence, les entrées $X$ et $Y$ ne sont qu'une seule et même séquence (on souhaite comparer les éléments de la séquence deux à deux) : on parle alors de _self-attention_.

Cela signifie que chaque position de la séquence $X$ "regarde" toutes les autres positions de cette même séquence pour construire sa propre représentation.

## Multi-head attention

En pratique, dans la plupart des modèles, le mécanisme d'attention est dupliqué plusieurs fois (avec des poids différents) et leurs sorties sont concaténées : on parle alors de _multi-head attention_.
Cela permet, d'une part, à chaque _head_ de se focaliser sur des aspects différents de la séquence (syntaxe, structure, position, etc.). Au final, cela permet une modélisation plus riche des dépendances.

## Schéma général

```{figure} ../img/multihead.png
:name: fig-multihead

Schéma d’un bloc Transformer avec multi-head attention (source : HuggingFace).
```



## Résumé

* Le mécanisme d'attention permet de capturer les dépendances entre éléments d'une séquence sans contrainte de distance.
* Il repose sur le calcul de similarité entre requêtes et clés, et la pondération des valeurs associées.
* Il est à la base des modèles Transformer, aujourd’hui omniprésents en NLP et en vision.