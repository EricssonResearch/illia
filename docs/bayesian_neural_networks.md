# Bayesian Neural Networks

## Introduction

**Bayesian Neural Networks (BNNs)** are a type of neural network that incorporates
Bayesian inference to estimate the uncertainty in the model parameters. Unlike
traditional neural networks, which use point estimates for weights, BNNs learn a
distribution over the weights, allowing for a better understanding of the uncertainty
associated with the model's predictions.

## Bayesian inference

**Bayesian inference** is based on Bayes' theorem, which allows for updating the
probabilities of a model based on observed evidence. The theorem is expressed as:

$$
P(\theta | D) = \frac{P(D | \theta) P(\theta)}{P(D)}
$$

where:

- $P(\theta | D)$ is the posterior distribution of the parameters given the data.
- $P(D | \theta)$ is the likelihood of the data given the parameters.
- $P(\theta)$ is the prior distribution of the parameters.
- $P(D)$ is the marginal likelihood (evidence), which acts as a normalization factor.

## Bayesian Neural Networks

In a **Bayesian Neural Network (BNN)**, instead of having fixed weights \(w\), a
probability distribution over the weights \(P(w)\) is assumed. The goal is to compute
the posterior distribution over the weights given the data:

$$
P(w | D) = \frac{P(D | w) P(w)}{P(D)}
$$

Since computing the exact posterior is intractable for large models, approximate
inference techniques are used, such as:

- **Variational Inference (VI):** Approximates the posterior \(P(w | D)\) with a simpler
  distribution \(q(w)\) by minimizing the Kullback-Leibler (KL) divergence.
- **Markov Chain Monte Carlo (MCMC):** Uses sampling methods to approximate the
  posterior distribution.

## Loss function

Instead of minimizing a simple loss function (e.g., Mean Squared Error), BNNs optimize
the evidence lower bound (ELBO), which is expressed as:

$$
\mathcal{L} = \mathbb{E}_{q(w)} [\log P(D | w)] - KL(q(w) || P(w))
$$

This loss function balances the likelihood of the data under the approximate posterior
and the KL divergence between the approximate posterior and the prior, providing better
regularization for the model.

## Applications

Bayesian Neural Networks are particularly useful in scenarios where uncertainty
estimation is critical. Common applications include:

- **Medical Diagnosis:** They provide uncertainty estimates in disease predictions,
  which is crucial for clinical decision-making.
- **Autonomous Vehicles:** They enhance decision-making under uncertainty, such as in
  autonomous driving in dynamic and unpredictable environments.
- **Financial Forecasting:** BNNs help quantify risk in investment models, enabling more
  informed decision-making.
- **Robotics:** They are useful in adaptive control and reinforcement learning, where
  environmental conditions may vary unpredictably.

## Advantages

Bayesian Neural Networks offer several advantages over traditional neural networks:

- **Uncertainty Estimation:** They provide a formal estimate of the uncertainty
  associated with model predictions, which is vital in applications where the
  reliability of decisions is crucial.
- **Overfitting Prevention:** Regularization through the prior helps prevent
  overfitting, especially in models with limited data.
- **Robustness to Small Datasets:** BNNs are more robust when trained on small datasets
  since the prior distribution can guide the model in the absence of large amounts of
  data.

This probabilistic approach not only enhances the interpretability of models but also
facilitates their application in contexts of high uncertainty and critical
decision-making.
