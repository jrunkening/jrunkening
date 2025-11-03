+++
title = " RiVAE"
description = "Riemannian Geometric Generation via Epistemic Graphs"
weight = 20

tags = ["Generative Model", "VAE", "Manifold", "Geodesic", "Riemannian Geometry"]

[params]
    math = true
    Toc = true
+++

Instead of sampling from a fixed Gaussian prior \\(z \sim \mathcal{N}(0, I)\\), RiVAE generates data by navigating through the learned latent geometry.
A well-trained VAE on MNIST defines a 2D latent space, but rather than treating it as flat, I reinterpret it as a Riemannian-like manifold induced by the model’s own epistemic structure — namely, its local uncertainty and global density.

#### [Repository (private)](https://jingyu.tplinkdns.com/gitea/Experiment/rivae) {.centered-text}

The MNIST dataset in latent space:

![](/jrunkening/projects/rivae/assets/mnist_latent_space.png)

[**Interactive Visualization**](/projects/rivae/assets/mnist_latent_space.html)

## Epistemic Geometry in Latent Space

In the RiVAE framework, every latent coordinate \\(z\\) carries two key epistemic quantities:

- Local uncertainty \\(U(z)\\): the posterior variance, describing epistemic noise,
indicates whether the existence of the data on this coordinate is making sense.
- Global density \\(P(z)\\): the log-likelihood under a Gaussian Mixture Model fitted on the latent embeddings.

Together they define a surrogate line element:
\\[d\mathcal{l} = (1 + \alpha_u u(z) + \alpha_p p(z)) \|z_i - z_j\|_2\\]
Where \\(U\\) and \\(P\\) penalize motion toward uncertain or low-density regions.

This metric — termed UDLD (Uncertainty and Density-aware Latent Distance) — effectively bends the Euclidean latent space according to the model’s epistemic beliefs.

## From Metric to Graph

A k-nearest neighbor (kNN) graph built from UDLD distances discretely approximates the manifold’s chart structure.
Edges implicitly align with geodesics because the UDLD penalization discourages connections that cross uncertain or low-density regions — effectively tracing along the manifold’s high-confidence surface.

![](/jrunkening/projects/rivae/assets/mnist_latent_space_manifold.png)

[**Interactive Visualization**](/projects/rivae/assets/mnist_latent_space_manifold.html)

However, unlike deterministic geodesic solvers, this graph-based construction enables non-deterministic trajectory generation.
At each step, the model can probabilistically select the next node among the top-k neighbors, with transition probabilities biased by edge weights or curvature penalties.
Thus, trajectories through latent space become stochastic realizations of geodesic motion — a diffusion-like walk across epistemic geometry, not merely interpolation.

## Local Approximation over Global Computation

Because full pairwise UDLD computation scales as \\(O(N^2)\\), direct distance matrices are infeasible for large datasets.
Instead, RiVAE employs a sampled-batch approximation, computing local UDLD neighborhoods within limited subsets while always including the start and target points.
This is not merely a computational trick — it echoes how continuous geodesics are integrated locally:
dense local information near the current region of interest suffices to maintain global consistency of the path.

## Geometric Generation

Once the kNN graph is constructed, generation proceeds as a geodesic walk between latent points.
Each intermediate latent code is decoded into an image, yielding a smooth semantic trajectory across the manifold.
The resulting animation reveals a discrete geodesic on the epistemic manifold —
each frame a valid MNIST digit, transitioning smoothly in semantic space without leaving the data support.

![](/jrunkening/projects/rivae/assets/mnist_latent_space_trajectory.png)

[**Interactive Visualization**](/projects/rivae/assets/mnist_latent_space_trajectory.html)

| Start | End | Transition |
|-------|-----| -----------|
| ![](/jrunkening/projects/rivae/assets/mnist_trajectory_start.png) | ![](/jrunkening/projects/rivae/assets/mnist_trajectory_end.png) | ![](/jrunkening/projects/rivae/assets/mnist_trajectory.gif) |

Such non-deterministic paths can represent families of valid interpolations, sampling multiple epistemically consistent routes between two data modes.

## Outlook

RiVAE transforms a conventional VAE into a geometric generator, embedding epistemic awareness directly into the latent topology.
Beyond smooth generation, this opens possibilities for:

- Curvature-aware latent regularization
- Geodesic gradient flows
- Uncertainty-weighted exploration in diffusion models.

Future work may generalize this to higher-dimensional latent spaces by learning Riemannian metrics directly via pullback Jacobians or graph Laplacians — bridging epistemic geometry and intrinsic generative control.
