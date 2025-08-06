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
# Attention Mechanism

In many deep learning contexts (machine translation, text summarization, sequence processing), models must handle variable-length inputs and focus on certain parts more than others.

The **attention** mechanism allows the model to give more weight to certain elements of a sequence when computing an output, depending on their **relevance**.

## Motivation

Consider the following sentence:

> _"An apple that had been on the tree in the garden for weeks had finally been picked up."_

which in French could be translated as:

> _"Une pomme qui était sur l'arbre du jardin depuis des semaines avait finalement été ramassée."_

Here, to correctly spell the word _ramassée_, one must be aware that it refers to the noun _une pomme_, which is feminine.

For a machine translation model to spell this word correctly, it must be able to model **long-range dependencies** between words.
However, classic **recurrent** or **convolutional** architectures struggle to efficiently handle these dependencies due to:
- the **bottleneck** in representations,
- the difficulty of memorizing distant information.

Attention addresses this limitation by allowing the model to **dynamically focus** on certain inputs when producing an output.

## General Principle

Instead of summarizing the input with a single fixed vector, as in classic recurrent encoders, attention generates an output by **weighting the different parts of the input** according to their relevance.

For each output element, the model performs a **weighted aggregation** of the input elements, where the weights reflect their **importance**.

## Metaphor: Queries, Keys, Values

Attention can be interpreted via the following metaphor:

- **Query (Q)**: what you are looking for
- **Key (K)**: what you have as reference
- **Value (V)**: what you extract

This mechanism is similar to what happens when manipulating a Python dictionary:
in a dictionary, you look for an exact key to get the associated value. Here, the query plays the role of the searched key, but instead of an exact match, you compare the query to all available keys (which are numerical vectors) by measuring their similarity (typically via a dot product).

Rather than retrieving the value of a single key, you perform a **weighted average** of the values associated with the keys most similar to the query. The weights of this average are precisely the similarities calculated between the query and each key.

## Mathematical Formulation

Let $X = [x_1, \dots, x_n]$ and $Y = [y_1, \dots, y_m]$ be two sequences of input vectors.
Attention consists in projecting $X$ into queries $Q$ and $Y$ into keys $K$ and values $V$:

\begin{align*}
Q &= XW^Q \\
K &= YW^K \\
V &= YW^V
\end{align*}

where $W^Q, W^K, W^V$ are learned weight matrices.

Attention is then defined by:

\begin{align*}
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V
\end{align*}

where $d_k$ is the dimension of the key vectors (used to stabilize training).

```{code-cell} ipython3
import torch
import numpy as np
import torch.nn.functional as F

torch.manual_seed(0)

Q = torch.randn(1, 4, 8)  # batch, length, dim
K = torch.randn(1, 6, 8)  # keys may have a different length
V = torch.randn(1, 6, 10) # values have the same length as keys, but can have a different dim

scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(8)
weights = F.softmax(scores, dim=-1)
output = torch.matmul(weights, V)

output.shape  # (1, 4, 10)
```

## Self-attention

In some cases, such as sequence processing, the inputs $X$ and $Y$ are the same sequence (we want to compare the elements of the sequence pairwise): this is called _self-attention_.

This means that each position in the sequence $X$ "looks at" all other positions in that same sequence to build its own representation.

## Multi-head attention

In practice, in most models, the attention mechanism is duplicated several times (with different weights) and their outputs are concatenated: this is called _multi-head attention_.
This allows each head to focus on different aspects of the sequence (syntax, structure, position, etc.), resulting in a richer modeling of dependencies.

## General diagram

```{figure} ../img/multihead.png
:name: fig-multihead

Diagram of a Transformer block with multi-head attention (source: HuggingFace).
```

## Summary

* The attention mechanism allows capturing dependencies between elements of a sequence without distance constraints.
* It relies on computing similarity between queries and keys, and weighting the associated values.
* It is the foundation of Transformer models, now ubiquitous in NLP and vision.
