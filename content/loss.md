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

In this section, we will present two standard losses, that are the mean squared error (that is mainly used for regression) and logistic loss (that is mostly used in classification settings).

## Mean Squared Error (MSE)

## Logistic loss

**TODO: ici, parler de non convexité, idéalement illustrer (peut-être sur Iris ?)**

## Loss regularizers
