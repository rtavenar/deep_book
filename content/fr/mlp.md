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

(sec:mlp)=
# Perceptrons multicouches

Dans le chapitre précédent, nous avons vu un modèle très simple appelé le perceptron.
Dans ce modèle, la sortie prédite $\hat{y}$ est calculée comme une combinaison linéaire des caractéristiques d'entrée plus un biais :

$$\hat{y} = \sum_{j=1}^d x_j w_j + b$$

En d'autres termes, nous optimisions parmi la famille des modèles linéaires, qui est une famille assez restreinte.

## Empiler des couches pour une meilleure expressivité

Afin de couvrir un plus large éventail de modèles, on peut empiler des neurones organisés en couches pour former un modèle plus complexe, comme le modèle ci-dessous, qui est appelé modèle à une couche cachée, car une couche supplémentaire de neurones est introduite entre les entrées et la sortie :

```{tikz}
    \node[text width=3cm, align=center] (in_title) at  (0, 6) {Couche d'entrée\\ $\mathbf{x}$};
    \node[text width=3cm, align=center] (h1_title) at  (3, 6) {Couche cachée\\ $\mathbf{h^{(1)}}$};
    \node[text width=3cm, align=center] (out_title) at  (6, 6) {Couche de sortie\\ $\mathbf{\hat{y}}$};

    \node[draw, circle, fill=blue, minimum size=17pt,inner sep=0pt] (in0) at  (0, 4) {};
    \node[draw, circle, fill=blue, minimum size=17pt,inner sep=0pt] (in1) at  (0, 3) {};
    \node[draw, circle, fill=blue, minimum size=17pt,inner sep=0pt] (in2) at  (0, 2) {};
    \node[draw, circle, fill=blue, minimum size=17pt,inner sep=0pt] (in3) at  (0, 1) {};
    \node[draw, circle, fill=blue, minimum size=17pt,inner sep=0pt] (in4) at  (0, 0) {};

    \node[draw, circle, minimum size=17pt,inner sep=0pt] (h1_0) at  (3, 5) {};
    \node[draw, circle, minimum size=17pt,inner sep=0pt] (h1_1) at  (3, 4) {};
    \node[draw, circle, minimum size=17pt,inner sep=0pt] (h1_2) at  (3, 3) {};
    \node[draw, circle, minimum size=17pt,inner sep=0pt] (h1_3) at  (3, 2) {};
    \node[draw, circle, minimum size=17pt,inner sep=0pt] (h1_4) at  (3, 1) {};
    \node[draw, circle, minimum size=17pt,inner sep=0pt] (h1_5) at  (3, 0) {};
    \node[draw, circle, minimum size=17pt,inner sep=0pt] (h1_6) at  (3, -1) {};
    
    \node[draw, circle, fill=green, minimum size=17pt,inner sep=0pt] (out_0) at  (6, 2) {};
    \draw[->] (in0) -- (h1_0);
    \draw[->] (in0) -- (h1_1);
    \draw[->] (in0) -- (h1_2);
    \draw[->] (in0) -- (h1_3);
    \draw[->] (in0) -- (h1_4);
    \draw[->] (in0) -- (h1_5);
    \draw[->] (in0) -- (h1_6);
    \draw[->] (in1) -- (h1_0);
    \draw[->] (in1) -- (h1_1);
    \draw[->] (in1) -- (h1_2);
    \draw[->] (in1) -- (h1_3);
    \draw[->] (in1) -- (h1_4);
    \draw[->] (in1) -- (h1_5);
    \draw[->] (in1) -- (h1_6);
    \draw[->] (in2) -- (h1_0);
    \draw[->] (in2) -- (h1_1);
    \draw[->] (in2) -- (h1_2);
    \draw[->] (in2) -- (h1_3);
    \draw[->] (in2) -- (h1_4);
    \draw[->] (in2) -- (h1_5);
    \draw[->] (in2) -- (h1_6);
    \draw[->] (in3) -- (h1_0);
    \draw[->] (in3) -- (h1_1);
    \draw[->] (in3) -- (h1_2);
    \draw[->] (in3) -- (h1_3);
    \draw[->] (in3) -- (h1_4);
    \draw[->] (in3) -- (h1_5);
    \draw[->] (in3) -- (h1_6);
    \draw[->] (in4) -- (h1_0);
    \draw[->] (in4) -- (h1_1);
    \draw[->] (in4) -- (h1_2);
    \draw[->] (in4) -- (h1_3);
    \draw[->] (in4) -- (h1_4);
    \draw[->] (in4) -- (h1_5);
    \draw[->] (in4) -- (h1_6);
    \draw[->] (h1_0) -- (out_0);
    \draw[->] (h1_1) -- (out_0);
    \draw[->] (h1_2) -- (out_0);
    \draw[->] (h1_3) -- (out_0);
    \draw[->] (h1_4) -- (out_0);
    \draw[->] (h1_5) -- (out_0);
    \draw[->] (h1_6) -- (out_0);


    \node[fill=white] (beta0) at  (1.5, 2) {$\mathbf{w^{(0)}}$};
    \node[fill=white] (beta1) at  (4.5, 2) {$\mathbf{w^{(1)}}$};
```

La question que l'on peut se poser maintenant est de savoir si cette couche cachée supplémentaire permet effectivement de couvrir une plus grande famille de modèles.
C'est à cela que sert le théorème d'approximation universelle ci-dessous.

```{admonition} Théorème d'approximation universelle

Le théorème d'approximation universelle stipule que toute fonction continue définie sur un ensemble compact peut être 
approchée d'aussi près que l'on veut par un réseau neuronal à une couche cachée avec activation sigmoïde.
```

En d'autres termes, en utilisant une couche cachée pour mettre en correspondance les entrées et les sorties, on peut maintenant approximer n'importe quelle fonction continue, ce qui est une propriété très intéressante.
Notez cependant que le nombre de neurones cachés nécessaire pour obtenir une qualité d'approximation donnée n'est pas discuté ici.
De plus, il n'est pas suffisant qu'une telle bonne approximation existe, une autre question importante est de savoir si les algorithmes d'optimisation que nous utiliserons convergeront _in fine_ vers cette solution ou non, ce qui n'est pas garanti, comme discuté plus en détail dans [le chapitre dédié](sec:sgd).

En pratique, nous observons empiriquement que pour atteindre une qualité d'approximation donnée, il est plus efficace (en termes de nombre de paramètres requis) d'empiler plusieurs couches cachées plutôt que de s'appuyer sur une seule :

```{tikz}
    \node[text width=3cm, align=center] (in_title) at  (0, 6) {Couche d'entrée\\ $\mathbf{x}$};
    \node[text width=3cm, align=center] (h1_title) at  (3, 6) {Première couche cachée\\ $\mathbf{h^{(1)}}$};
    \node[text width=3cm, align=center] (h1_title) at  (6, 6) {Seconde couche cachée\\ $\mathbf{h^{(2)}}$};
    \node[text width=3cm, align=center] (out_title) at  (9, 6) {Couche de sortie\\ $\mathbf{\hat{y}}$};

    \node[draw, circle, fill=blue, minimum size=17pt,inner sep=0pt] (in0) at  (0, 4) {};
    \node[draw, circle, fill=blue, minimum size=17pt,inner sep=0pt] (in1) at  (0, 3) {};
    \node[draw, circle, fill=blue, minimum size=17pt,inner sep=0pt] (in2) at  (0, 2) {};
    \node[draw, circle, fill=blue, minimum size=17pt,inner sep=0pt] (in3) at  (0, 1) {};
    \node[draw, circle, fill=blue, minimum size=17pt,inner sep=0pt] (in4) at  (0, 0) {};

    \node[draw, circle, fill=cyan, minimum size=17pt,inner sep=0pt] (h1_0) at  (3, 5) {};
    \node[draw, circle, fill=cyan, minimum size=17pt,inner sep=0pt] (h1_1) at  (3, 4) {};
    \node[draw, circle, fill=cyan, minimum size=17pt,inner sep=0pt] (h1_2) at  (3, 3) {};
    \node[draw, circle, fill=cyan, minimum size=17pt,inner sep=0pt] (h1_3) at  (3, 2) {};
    \node[draw, circle, fill=cyan, minimum size=17pt,inner sep=0pt] (h1_4) at  (3, 1) {};
    \node[draw, circle, fill=cyan, minimum size=17pt,inner sep=0pt] (h1_5) at  (3, 0) {};
    \node[draw, circle, fill=cyan, minimum size=17pt,inner sep=0pt] (h1_6) at  (3, -1) {};

    \node[draw, circle, fill=teal, minimum size=17pt,inner sep=0pt] (h2_0) at  (6, 5) {};
    \node[draw, circle, fill=teal, minimum size=17pt,inner sep=0pt] (h2_1) at  (6, 4) {};
    \node[draw, circle, fill=teal, minimum size=17pt,inner sep=0pt] (h2_2) at  (6, 3) {};
    \node[draw, circle, fill=teal, minimum size=17pt,inner sep=0pt] (h2_3) at  (6, 2) {};
    \node[draw, circle, fill=teal, minimum size=17pt,inner sep=0pt] (h2_4) at  (6, 1) {};
    \node[draw, circle, fill=teal, minimum size=17pt,inner sep=0pt] (h2_5) at  (6, 0) {};
    \node[draw, circle, fill=teal, minimum size=17pt,inner sep=0pt] (h2_6) at  (6, -1) {};
    
    \node[draw, circle, fill=green, minimum size=17pt,inner sep=0pt] (out_0) at  (9, 2) {};
    \draw[->] (in0) -- (h1_0);
    \draw[->] (in0) -- (h1_1);
    \draw[->] (in0) -- (h1_2);
    \draw[->] (in0) -- (h1_3);
    \draw[->] (in0) -- (h1_4);
    \draw[->] (in0) -- (h1_5);
    \draw[->] (in0) -- (h1_6);
    \draw[->] (in1) -- (h1_0);
    \draw[->] (in1) -- (h1_1);
    \draw[->] (in1) -- (h1_2);
    \draw[->] (in1) -- (h1_3);
    \draw[->] (in1) -- (h1_4);
    \draw[->] (in1) -- (h1_5);
    \draw[->] (in1) -- (h1_6);
    \draw[->] (in2) -- (h1_0);
    \draw[->] (in2) -- (h1_1);
    \draw[->] (in2) -- (h1_2);
    \draw[->] (in2) -- (h1_3);
    \draw[->] (in2) -- (h1_4);
    \draw[->] (in2) -- (h1_5);
    \draw[->] (in2) -- (h1_6);
    \draw[->] (in3) -- (h1_0);
    \draw[->] (in3) -- (h1_1);
    \draw[->] (in3) -- (h1_2);
    \draw[->] (in3) -- (h1_3);
    \draw[->] (in3) -- (h1_4);
    \draw[->] (in3) -- (h1_5);
    \draw[->] (in3) -- (h1_6);
    \draw[->] (in4) -- (h1_0);
    \draw[->] (in4) -- (h1_1);
    \draw[->] (in4) -- (h1_2);
    \draw[->] (in4) -- (h1_3);
    \draw[->] (in4) -- (h1_4);
    \draw[->] (in4) -- (h1_5);
    \draw[->] (in4) -- (h1_6);

    \draw[->] (h1_0) -- (h2_0);
    \draw[->] (h1_1) -- (h2_0);
    \draw[->] (h1_2) -- (h2_0);
    \draw[->] (h1_3) -- (h2_0);
    \draw[->] (h1_4) -- (h2_0);
    \draw[->] (h1_5) -- (h2_0);
    \draw[->] (h1_6) -- (h2_0);
    \draw[->] (h1_0) -- (h2_1);
    \draw[->] (h1_1) -- (h2_1);
    \draw[->] (h1_2) -- (h2_1);
    \draw[->] (h1_3) -- (h2_1);
    \draw[->] (h1_4) -- (h2_1);
    \draw[->] (h1_5) -- (h2_1);
    \draw[->] (h1_6) -- (h2_1);
    \draw[->] (h1_0) -- (h2_2);
    \draw[->] (h1_1) -- (h2_2);
    \draw[->] (h1_2) -- (h2_2);
    \draw[->] (h1_3) -- (h2_2);
    \draw[->] (h1_4) -- (h2_2);
    \draw[->] (h1_5) -- (h2_2);
    \draw[->] (h1_6) -- (h2_2);
    \draw[->] (h1_0) -- (h2_3);
    \draw[->] (h1_1) -- (h2_3);
    \draw[->] (h1_2) -- (h2_3);
    \draw[->] (h1_3) -- (h2_3);
    \draw[->] (h1_4) -- (h2_3);
    \draw[->] (h1_5) -- (h2_3);
    \draw[->] (h1_6) -- (h2_3);
    \draw[->] (h1_0) -- (h2_4);
    \draw[->] (h1_1) -- (h2_4);
    \draw[->] (h1_2) -- (h2_4);
    \draw[->] (h1_3) -- (h2_4);
    \draw[->] (h1_4) -- (h2_4);
    \draw[->] (h1_5) -- (h2_4);
    \draw[->] (h1_6) -- (h2_4);
    \draw[->] (h1_0) -- (h2_5);
    \draw[->] (h1_1) -- (h2_5);
    \draw[->] (h1_2) -- (h2_5);
    \draw[->] (h1_3) -- (h2_5);
    \draw[->] (h1_4) -- (h2_5);
    \draw[->] (h1_5) -- (h2_5);
    \draw[->] (h1_6) -- (h2_5);
    \draw[->] (h1_0) -- (h2_6);
    \draw[->] (h1_1) -- (h2_6);
    \draw[->] (h1_2) -- (h2_6);
    \draw[->] (h1_3) -- (h2_6);
    \draw[->] (h1_4) -- (h2_6);
    \draw[->] (h1_5) -- (h2_6);
    \draw[->] (h1_6) -- (h2_6);

    \draw[->] (h2_0) -- (out_0);
    \draw[->] (h2_1) -- (out_0);
    \draw[->] (h2_2) -- (out_0);
    \draw[->] (h2_3) -- (out_0);
    \draw[->] (h2_4) -- (out_0);
    \draw[->] (h2_5) -- (out_0);
    \draw[->] (h2_6) -- (out_0);


    \node[fill=white] (beta0) at  (1.5, 2) {$\mathbf{w^{(0)}}$};
    \node[fill=white] (beta1) at  (4.5, 2) {$\mathbf{w^{(1)}}$};
    \node[fill=white] (beta2) at  (7.5, 2) {$\mathbf{w^{(2)}}$};
```

La représentation graphique ci-dessus correspond au modèle suivant :

\begin{align}
  {\color[rgb]{0,1,0}\hat{y}} &= \varphi_\text{out} \left( \sum_i w^{(2)}_{i} {\color{teal}h^{(2)}_{i}} + b^{(2)} \right) \\
  \forall i, {\color{teal}h^{(2)}_{i}} &= \varphi \left( \sum_j w^{(1)}_{ij} {\color[rgb]{0.16,0.61,0.91}h^{(1)}_{j}} + b^{(1)}_{i} \right) \\
  \forall i, {\color[rgb]{0.16,0.61,0.91}h^{(1)}_{i}} &= \varphi \left( \sum_j w^{(0)}_{ij} {\color{blue}x_{j}} + b^{(0)}_{i} \right)
  \label{eq:mlp_2hidden}
\end{align}

Pour être précis, les termes de biais $b^{(l)}_i$ ne sont pas représentés dans la représentation graphique ci-dessus.

De tels modèles avec une ou plusieurs couches cachées sont appelés **Perceptrons multicouches** (ou _Multi-Layer Perceptrons_, MLP).

## Décider de l'architecture d'un MLP

Lors de la conception d'un modèle de perceptron multicouche destiné à être utilisé pour un problème spécifique, certaines quantités sont fixées par le problème en question et d'autres sont des hyper-paramètres du modèle.

Prenons l'exemple du célèbre jeu de données de classification d'iris :

```{code-cell}
import pandas as pd

iris = pd.read_csv("../data/iris.csv", index_col=0)
iris
```

L'objectif ici est d'apprendre à déduire l'attribut "cible" (3 classes différentes possibles) à partir des informations contenues dans les 4 autres attributs.

La structure de ce jeu de données dicte :
* le nombre de neurones dans la couche d'entrée, qui est égal au nombre d'attributs descriptifs dans notre jeu de données (ici, 4), et
* le nombre de neurones dans la couche de sortie, qui est ici égal à 3, puisque le modèle est censé produire une probabilité par classe cible.

De manière plus générale, pour la couche de sortie, on peut être confronté à plusieurs situations :
* lorsqu'il s'agit de régression, le nombre de neurones de la couche de sortie est égal au nombre de caractéristiques à prédire par le modèle,
* quand il s'agit de classification
  * Dans le cas d'une classification binaire, le modèle aura un seul neurone de sortie qui indiquera la probabilité de la classe positive,
  * dans le cas d'une classification multi-classes, le modèle aura autant de neurones de sortie que le nombre de classes du problème.

Une fois que ces nombres de neurones d'entrée / sortie sont fixés, le nombre de neurones cachés ainsi que le nombre de neurones par couche cachée restent des hyper-paramètres du modèle.

## Fonctions d'activation

Un autre hyper-paramètre important des réseaux neuronaux est le choix de la fonction d'activation $\varphi$.

Il est important de noter que si nous utilisons la fonction identité comme fonction d'activation, quelle que soit la profondeur de notre MLP, nous ne couvrirons plus que la famille des modèles linéaires.
En pratique, nous utiliserons donc des fonctions d'activation qui ont un certain régime linéaire mais qui ne se comportent pas comme une fonction linéaire sur toute la gamme des valeurs d'entrée.

Historiquement, les fonctions d'activation suivantes ont été proposées :


\begin{align*}
    \text{tanh}(x) =& \frac{2}{1 + e^{-2x}} - 1 \\
    \text{sigmoid}(x) =& \frac{1}{1 + e^{-x}} \\
    \text{ReLU}(x) =& \begin{cases}
                        x \text{ si } x \gt 0\\
                        0 \text{ sinon }
                      \end{cases}
\end{align*}

```{code-cell} ipython3
:tags: [hide-input]

import numpy as np

%config InlineBackend.figure_format = 'svg'
%matplotlib inline
import matplotlib.pyplot as plt
from notebook_utils import prepare_notebook_graphics
prepare_notebook_graphics()

def tanh(x):
    return 2. / (1. + np.exp(-2 * x)) - 1.

def sigmoid(x):
    return 1. / (1. + np.exp(-x))

def relu(x):
    y = x.copy()
    y[y < 0] = 0.
    return y

x = np.linspace(-4, 4, 50)

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.plot(x, tanh(x))
plt.grid('on')
plt.ylim([-1.1, 4.1])
plt.title("tanh")

plt.subplot(1, 3, 2)
plt.plot(x, sigmoid(x))
plt.grid('on')
plt.ylim([-1.1, 4.1])
plt.title("sigmoid")

plt.subplot(1, 3, 3)
plt.plot(x, relu(x))
plt.grid('on')
plt.ylim([-1.1, 4.1])
plt.title("ReLU");
```

En pratique, la fonction ReLU (et certaines de ses variantes) est la plus utilisée de nos jours, pour des raisons qui seront discutées plus en détail dans [notre chapitre consacré à l'optimisation](sec:sgd).

### Le cas particulier de la couche de sortie

Vous avez peut-être remarqué que dans la formulation du MLP fournie par l'équation (1), la couche de sortie possède sa propre fonction d'activation, notée $\varphi_\text{out}$.
Cela s'explique par le fait que le choix de la fonction d'activation pour la couche de sortie d'un réseau neuronal est spécifique au problème à résoudre.

En effet, vous avez pu constater que les fonctions d'activation abordées dans la section précédente ne partagent pas la même plage de valeurs de sortie.
Il est donc primordial de choisir une fonction d'activation adéquate pour la couche de sortie, de sorte que notre modèle produise des valeurs cohérentes avec les quantités qu'il est censé prédire.

Si, par exemple, notre modèle est censé être utilisé dans l'ensemble de données sur les logements de Boston dont nous avons parlé [dans le chapitre précédent](sec:boston), l'objectif est de prédire les prix des logements, qui sont censés être des quantités non négatives.
Il serait donc judicieux d'utiliser ReLU (qui peut produire toute valeur positive) comme fonction d'activation pour la couche de sortie dans ce cas.

Comme indiqué précédemment, dans le cas de la classification binaire, le modèle aura un seul neurone de sortie et ce neurone produira la probabilité associée à la classe positive.
Cette quantité devra se situer dans l'intervalle $[0, 1]$, et la fonction d'activation sigmoïde est alors le choix par défaut dans ce cas.

Enfin, lorsque la classification multi-classes est en jeu, nous avons un neurone par classe de sortie et chaque neurone est censé fournir la probabilité pour une classe donnée.
Dans ce contexte, les valeurs de sortie doivent être comprises entre 0 et 1, et leur somme doit être égale à 1.
À cette fin, nous utilisons la fonction d'activation softmax définie comme suit :

$$
  \forall i, \text{softmax}(o_i) = \frac{e^{o_i}}{\sum_j e^{o_j}}
$$

où, pour tous les $i$, les $o_i$ sont les valeurs des neurones de sortie avant application de la fonction d'activation.

## Déclarer un MLP en `keras`

Pour définir un modèle MLP dans `keras`, il suffit d'empiler des couches.
A titre d'exemple, si l'on veut coder un modèle composé de :
* une couche d'entrée avec 10 neurones,
* d'une couche cachée de 20 neurones avec activation ReLU,
* une couche de sortie composée de 3 neurones avec activation softmax, 

le code sera le suivant :

```{code-cell}
:tags: [remove-stderr]

import keras
from keras.layers import Dense, InputLayer
from keras.models import Sequential

model = Sequential([
    InputLayer(input_shape=(10, )),
    Dense(units=20, activation="relu"),
    Dense(units=3, activation="softmax")
])

model.summary()
```

Notez que `model.summary()` fournit un aperçu intéressant d'un modèle défini et de ses paramètres.

````{admonition} Exercice #1

En vous basant sur ce que nous avons vu dans ce chapitre, pouvez-vous expliquer le nombre de paramètres retournés par `model.summary()` ci-dessus ?

```{admonition} Solution
:class: dropdown, tip

Notre couche d'entrée est composée de 10 neurones, et notre première couche est entièrement connectée, donc chacun de ces neurones est connecté à un neurone de la couche cachée par un paramètre, ce qui fait déjà $10 \times 20 = 200$ paramètres.
De plus, chacun des neurones de la couche cachée possède son propre paramètre de biais, ce qui fait $20$ paramètres supplémentaires.
Nous avons donc 220 paramètres, tels que sortis par `model.summary()` pour la couche `"dense (Dense)"`.

De la même manière, pour la connexion des neurones de la couche cachée à ceux de la couche de sortie, le nombre total de paramètres est de $20 \times 3 = 60$ pour les poids plus $3$ paramètres supplémentaires pour les biais.

Au total, nous avons $220 + 63 = 283$ paramètres dans ce modèle.
```
````

`````{admonition} Exercice #2

Déclarez, en `keras`, un MLP avec une couche cachée composée de 100 neurones et une activation ReLU pour le jeu de données Iris présenté ci-dessus.

````{admonition} Solution
:class: dropdown, tip

```python
model = Sequential([
    InputLayer(input_shape=(4, )),
    Dense(units=100, activation="relu"),
    Dense(units=3, activation="softmax")
])
```
````
`````

`````{admonition} Exercice #3

Même question pour le jeu de données sur le logement à Boston présenté ci-dessous (le but ici est de prédire l'attribut `PRICE` en fonction des autres).

````{admonition} Solution
:class: dropdown, tip

```python
model = Sequential([
    InputLayer(input_shape=(6, )),
    Dense(units=100, activation="relu"),
    Dense(units=1, activation="relu")
])
```
````
`````


```{code-cell}
:tags: [hide-input]

boston = pd.read_csv("../data/boston.csv")[["RM", "CRIM", "INDUS", "NOX", "AGE", "TAX", "PRICE"]]
boston
```