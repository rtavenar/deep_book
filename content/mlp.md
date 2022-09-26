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

**TODO intro**

## Why stacking layers?

In the previous chapter, we have seen a very simple model called the Perceptron.
In this model, the predicted output $\hat{y}$ is computed as a linear combination of the input features plus a bias:

$$\hat{y} = \sum_{j=1}^d x_j w_j + b$$

In other words, we were optimizing among the family of linear models, which is a quite restricted family.
In order to cover a wider range of models, one can stack neurons organized in layers to form a more complex model, such as the model below, which is called a one-hidden-layer-model, since an extra layer of neurons is introduced between the inputs and the output:

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
    \draw[->] (h1_5) -- (out_0);
    \draw[->] (h1_6) -- (out_0);


    \node[fill=white] (beta0) at  (1.5, 2) {$\mathbf{w^{(0)}}$};
    \node[fill=white] (beta1) at  (4.5, 2) {$\mathbf{w^{(1)}}$};
```

So, now we should ask: does this help us cover a wider family of models?
Here, the answer is that it depends on the activation function we use for the hidden layer.
**TODO: detail that**, starting with the identity case and showing that it does nothing more than what a Perceptron would do, and then introducing the sigmoid function and the UAT.


```{admonition} Universal Approximation Theorem

The Universal Approximation Theorem states that any continuous function defined on a compact set can be 
approximated as closely as one wants by a one-hidden-layer neural network with sigmoid activation.
```

This Universal Approximation Theorem shows that by using a hidden layer to map inputs to outputs, one can now approximate any continuous function, which is a very interesting property.



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
