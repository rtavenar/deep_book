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
# Multi Layer Perceptrons (MLP)

In the previous chapter, we have seen a very simple model called the Perceptron.
In this model, the predicted output $\hat{y}$ is computed as a linear combination of the input features plus a bias:

$$\hat{y} = \sum_{j=1}^d x_j w_j + b$$

In other words, we were optimizing among the family of linear models, which is a quite restricted family.
In order to cover a wider range of models, one can stack neurons organized in layers to form a more complex model, such as the model below, which is called a one-hidden-layer model, since an extra layer of neurons is introduced between the inputs and the output:

```{tikz}
    \node[text width=3cm, align=center] (in_title) at  (0, 6) {Input layer\\ $\mathbf{x}$};
    \node[text width=3cm, align=center] (h1_title) at  (3, 6) {Hidden layer 1\\ $\mathbf{h^{(1)}}$};
    \node[text width=3cm, align=center] (out_title) at  (6, 6) {Output layer\\ $\mathbf{\hat{y}}$};

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

The question one might ask now is whether this added hidden layer effectively allows to cover a wider family of models.
This is what the Universal Approximation Theorem below is all about.

```{admonition} Universal Approximation Theorem

The Universal Approximation Theorem states that any continuous function defined on a compact set can be 
approximated as closely as one wants by a one-hidden-layer neural network with sigmoid activation.
```

In other words, by using a hidden layer to map inputs to outputs, one can now approximate any continuous function, which is a very interesting property.
Note however that the number of hidden neurons that is necessary to achieve a given approximation quality is not discussed here.
Moreover, it is not sufficient that such a good approximation exists, another important question is whether the optimization algorithms we will use will eventually converge to this solution or not, which is not guaranteed, as discussed in more details in [the dedicqted chapter](sec:sgd).

In practice, we observe empirically that in order to achieve a given approximation quality, it is more efficient (in terms of the number of parameters required) to stack several hidden layers rather than rely on a single one :

```{tikz}
    \node[text width=3cm, align=center] (in_title) at  (0, 6) {Input layer\\ $\mathbf{x}$};
    \node[text width=3cm, align=center] (h1_title) at  (3, 6) {Hidden layer 1\\ $\mathbf{h^{(1)}}$};
    \node[text width=3cm, align=center] (h1_title) at  (6, 6) {Hidden layer 2\\ $\mathbf{h^{(1)}}$};
    \node[text width=3cm, align=center] (out_title) at  (9, 6) {Output layer\\ $\mathbf{\hat{y}}$};

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

    \node[draw, circle, minimum size=17pt,inner sep=0pt] (h2_0) at  (6, 5) {};
    \node[draw, circle, minimum size=17pt,inner sep=0pt] (h2_1) at  (6, 4) {};
    \node[draw, circle, minimum size=17pt,inner sep=0pt] (h2_2) at  (6, 3) {};
    \node[draw, circle, minimum size=17pt,inner sep=0pt] (h2_3) at  (6, 2) {};
    \node[draw, circle, minimum size=17pt,inner sep=0pt] (h2_4) at  (6, 1) {};
    \node[draw, circle, minimum size=17pt,inner sep=0pt] (h2_5) at  (6, 0) {};
    \node[draw, circle, minimum size=17pt,inner sep=0pt] (h2_6) at  (6, -1) {};
    
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

The above graphical representation corresponds to the following model:

\begin{align}
  \hat{y} &= \varphi \left( \sum_i w^{(2)}_{i} h^{(2)}_{i} + b^{(2)} \right) \\
  \forall i, h^{(2)}_{i} &= \varphi \left( \sum_j w^{(1)}_{ij} h^{(1)}_{j} + b^{(1)}_{i} \right) \\
  \forall i, h^{(1)}_{i} &= \varphi \left( \sum_j w^{(0)}_{ij} x_{j} + b^{(0)}_{i} \right)
\end{align}

To be even more precise, the bias terms $b^{(l)}_i$ are not represented in the graphical representation above.

Such models with one or more hidden layers are called **Multi Layer Perceptrons** and we will present their characteristics in the following.



## Activation functions

## The special case of the output layer

## Declaring an MLP in `keras`

In order to define a MLP model in `keras`, one just has to stack layers.
As an example, if one wants to code a model made of:
* an input layer with 10 neurons,
* a hidden layer made of 20 neurons with ReLU activation,
* an output layer made of 3 neurons with softmax activation, 

the code will look like:

```{code-cell}

from tensorflow.keras.layers import Dense, InputLayer
from tensorflow.keras.models import Sequential

model = Sequential([
    InputLayer(input_shape=(10, )),
    Dense(units=20, activation="relu"),
    Dense(units=3, activation="softmax")
])

model.summary()
```

Note that `model.summary()` provides an interesting overview of a defined model and its parameters.

````{admonition} Exercise

Relying on what we have seen in this chapter, can you explain the number of parameters returned by `model.summary()` above?

```{admonition} Solution
:class: dropdown, tip

Our input layer is made of 10 neurons, and our first layer is fully connected, hence each of these neurons is connected to a neuron in the hidden layer through a parameter, which already makes $10 \times 20 = 200$ parameters.
Moreover, each of the hidden layer neurons has its own bias parameter, which is $20$ more parameters.
We then have 220 parameters, as output by `model.summary()` for the layer `"dense (Dense)"`.

Similarly, for the connection of the hidden layer neurons to those in the output layer, the total number of parameters is $20 \times 3 = 60$ for the weights plus $3$ extra parameters for the biases.

Overall, we have $220 + 63 = 283$ parameters in this model.
```
````
