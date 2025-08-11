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

(sec:generative)=
# Generative Neural Networks

Generative models aim to learn the distribution of training data. This distribution can be estimated explicitly, by learning a parametric form of $p(x)$ or the conditional probability $p(x|y)$, or approached implicitly, without a closed form but allowing sampling of new data.

Among the main generative models, we find Gaussian Mixture Models (GMM), Variational Auto-Encoders (VAE), Generative Adversarial Networks (GAN), and diffusion models. Each of these models proposes a different approach to modeling and generating data, ranging from direct estimation of the distribution to more indirect methods based on sampling or competition between networks.

## Auto-encoders

Auto-encoders {cite:p}`hinton2006reducing` are networks that learn to compress information into a latent space. An auto-encoder consists of an Encoder block and a Decoder block, used as follows:

\begin{align}
z =& \text{Encoder}(x) \\
\hat{x} =& \text{Decoder}(z)
\end{align}

In other words, the encoder projects the input $x$ to a latent representation $z$, usually of lower dimension, and the decoder reconstructs an approximation $\hat{x}$ from $z$. This process can be seen as a generalization of PCA to the nonlinear case. However, a standard auto-encoder is not a generative model, as it does not impose any particular distribution on the latent space $z$ and thus does not provide a way to sample new data.

## Variational Auto-Encoders (VAE)

VAEs {cite:p}`kingma2014auto` turn the auto-encoder into a generative model by imposing a prior on the latent variable $z$, typically a normal distribution $z \sim \mathcal{N}(0, I)$. A penalty, in the form of Kullback-Leibler (KL) divergence, is added to the loss function to encourage the latent distribution to match this prior.

To generate new data:
1. sample $z$ from $\mathcal{N}(0, I)$
2. compute $x_\text{gen} = \text{Decoder}(z)$

## Generative Adversarial Networks (GAN)

Proposed by {cite:p}`goodfellow2014generative`, GANs train two networks:

- a Generator $G$ that produces $x_{\text{fake}} = G(z)$ from noise $z \sim \mathcal{N}(0, I)$
- a Discriminator $D$ that predicts whether an input $x$ is real ($y=1$) or generated ($y=0$)

The optimized loss function is:

\begin{equation}
\mathbb{E}_{x \sim p_r}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log (1 - D(G(z)))]
\end{equation}

In practice, training alternates between updating $D$ (better discrimination, maximizing the loss) and updating $G$ (better generation, minimizing the loss).

For generation, as with a VAE, one samples $z \sim \mathcal{N}(0, I)$ and feeds it to the generator to produce a new sample $G(z)$.

In practice, GAN optimization is often unstable, and it is often necessary to use tricks to stabilize it (see Wasserstein GAN for example {cite:p}`arjovsky2017wasserstein`).

## Diffusion Models

Diffusion models, introduced by {cite}`ho2020denoising`, are based on an original idea: Gaussian noise is progressively added to the data, and a model is trained to reverse this process, i.e., to denoise the data step by step. During generation, one starts from pure noise and gradually transforms it into a realistic data sample.

## Conditional Flow Matching

Conditional Flow Matching, proposed by {cite}`lipman2023flow`, consists in learning a vector field that progressively transports samples from noise (initial state $t=0$, corresponding to the $z$ introduced above for VAEs and GANs) to real data (final state $t=1$, corresponding to $x$ above). Training relies on minimizing the following loss function:

\begin{equation}
\mathbb{E}_{x_0, x_1, t} \left[ u^\theta (x, t) - (x_1 - x_0) \right]
\end{equation}
where $x = t x_0 + (1 - t) x_1$.

Once the model $u^\theta$ is learned, generation is performed by solving a differential equation, for example with the Euler scheme, starting from a sample $x_0$ drawn from $\mathcal{N}(0, I)$:

\begin{equation}
x_{t+\varepsilon} \leftarrow x_t + \varepsilon u^\theta (x_t, t)
\end{equation}

This process can be seen as a guided interpolation between noise and data.

## Summary

In summary, generative models offer powerful tools for modeling and sampling data distributions. Depending on the chosen approach, they may compress information by imposing a probabilistic structure on the latent space (VAE), generate data through competition between networks (GAN), or produce samples via dynamic and progressive processes (diffusion and flow matching).

## References

```{bibliography}
:filter: docname in docnames
```
