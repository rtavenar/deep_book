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
# Réseaux neuronaux génératifs

Les modèles génératifs ont pour objectif d'apprendre la distribution des données d'entraînement. Cette distribution peut être estimée de façon explicite, en apprenant une forme paramétrique de $p(x)$ ou de la probabilité conditionnelle $p(x|y)$, ou bien approchée de manière implicite, sans forme close mais en permettant l'échantillonnage de nouvelles données.

Parmi les principaux modèles génératifs, on retrouve les modèles de mélange gaussiens (GMM), les auto-encodeurs variationnels (VAE), les réseaux adversaires génératifs (GAN) et les modèles de diffusion. Chacun de ces modèles propose une approche différente pour modéliser et générer des données, allant de l'estimation directe de la distribution à des méthodes plus indirectes basées sur l'échantillonnage ou la compétition entre réseaux.

## Auto-encodeurs

Les auto-encodeurs {cite:p}`hinton2006reducing` sont des réseaux qui apprennent à compresser l'information dans un espace latent. 
Un auto-encodeur est constitué d'un bloc Encodeur et d'un Bloc Décodeur, utilisés comme suit :

\begin{align}
z =& \text{Encodeur}(x) \\
\hat{x} =& \text{Decodeur}(z)
\end{align}

Autrement dit, un encodeur projette l'entrée $x$ vers une représentation latente $z$, généralement de plus faible dimension, puis un décodeur reconstruit une approximation $\hat{x}$ à partir de $z$. Ce fonctionnement peut être vu comme une généralisation de l'ACP au cas non linéaire. 
Toutefois, un auto-encodeur standard n'est pas un modèle génératif, car il n'impose pas de distribution particulière sur l'espace latent $z$ et n'offre donc pas de façon de tirer de nouveaux samples.

## Variational Auto-Encoders (VAE)

Les VAE {cite:p}`kingma2014auto` transforment l'auto-encodeur en modèle génératif en imposant un _a priori_ sur la variable latente $z$, typiquement une loi normale $z \sim \mathcal{N}(0, I)$. Une pénalisation, sous forme de divergence de Kullback-Leibler (KL), est ajoutée à la fonction de perte à optimiser pour encourager la distribution latente à respecter cet _a priori_. 

Pour générer de nouvelles données :
1. on tire un $z$ selon $\mathcal{N}(0, I)$
2. on calcule $x_\text{gen} = \text{Decodeur}(x)$

## Generative Adversarial Networks (GAN)

Proposés par {cite:p}`goodfellow2014generative`, les GAN entraînent deux réseaux :

- un Générateur $G$ qui produit $x_{\text{fake}} = G(z)$ à partir de bruit $z \sim \mathcal{N}(0, I)$
- un Discriminateur $D$ qui prédit si une entrée $x$ est réelle ($y=1$) ou générée ($y=0$)

La fonction de perte optimisée est la suivante :

\begin{equation}
\mathbb{E}_{x \sim p_r}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log (1 - D(G(z)))]
\end{equation}

Concrètement, l'entraînement alterne entre mise à jour de $D$ (meilleure discrimination, maximisation de la fonction de perte) et mise à jour de $G$ (meilleure génération, minimisation de la fonction de perte).

Pour la génération, comme pour un VAE, on tire un $z \sim \mathcal{N}(0, I)$ puis on le fournit en entrée au générateur pour générer un nouveau sample $G(z)$.

En pratique, l'optimisation d'un GAN est souvent instable, et il est souvent nécessaire d'utiliser des astuces pour le stabiliser (_cf_ les Wasserstein GAN par exemple {cite:p}`arjovsky2017wasserstein`).


## Modèles de diffusion

Les modèles de diffusion, introduits par {cite}`ho2020denoising`, reposent sur une idée originale : on ajoute progressivement du bruit gaussien aux données, puis on entraîne un modèle à inverser ce processus, c'est-à-dire à débruiter les données étape par étape. Lors de la génération, on part d'un bruit pur et on le transforme progressivement en une donnée réaliste.

## Conditional Flow Matching

Le Conditional Flow Matching, proposé par {cite}`lipman2023flow`, consiste à apprendre un champ de vecteurs qui transporte progressivement les échantillons du bruit (état initial $t=0$, correspondant au $z$ introduit plus haut pour les VAE et les GAN) vers les données réelles (état final $t=1$, correspodant au $x$ plus haut). L'entraînement repose sur la minimisation de la fonction de perte suivante :

\begin{equation}
\mathbb{E}_{x_0, x_1, t} \left[ u^\theta (x, t) - (x_1 - x_0) \right]
\end{equation}
où $x = t x_0 + (1 - t) x_1$.

Une fois le modèle $u^\theta$ appris, la génération s'effectue en résolvant une équation différentielle, par exemple avec le schéma d'Euler, en partant d'un sample $x_0$ tiré de $\mathcal{N}(0, I)$ :

\begin{equation}
x_{t+\varepsilon} \leftarrow x_t + \varepsilon u^\theta (x_t, t)
\end{equation}

Ce processus peut être vu comme une interpolation guidée entre le bruit et les données.

## Résumé

En résumé, les modèles génératifs offrent des outils puissants pour modéliser et échantillonner la distribution des données. Selon l'approche choisie, ils peuvent consister à compresser l'information en imposant une structure probabiliste sur l'espace latent (VAE), à générer des données par compétition entre réseaux (GAN), ou encore à produire des échantillons via des processus dynamiques et progressifs (diffusion et _flow matching_).

## Références

```{bibliography}
:filter: docname in docnames
```
