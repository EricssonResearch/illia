# Bayesian Neural Networks

**Bayesian Neural Networks (BNNs)** represent a paradigm that integrates Bayesian
inference into deep learning models. Unlike traditional neural networks, where parameters
(weights and biases) are fixed values determined through optimization algorithms like
backpropagation and gradient descent, BNNs model these parameters as probability
distributions. This conceptual shift allows capturing the inherent uncertainty in both
the model's parameters and its predictions, offering a more comprehensive understanding
of the model's limitations and reliability.

![image](https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fjonascleveland.com%2Fwp-content%2Fuploads%2F2023%2F07%2FBayesian-Network-vs-Neural-Network.png&f=1&nofb=1&ipt=79ec39d4258da81fe61c9d9395d92f984259b951150c23451b0892cd578e92e4)

## Theoretical Foundations of Bayesian Inference

Bayesian inference is based on **Bayes' Theorem**, which provides a mathematical
framework for updating beliefs about a model when new observations become available. To
understand this concept, it's helpful to consider the process of human learning:
initially, we have prior knowledge about a phenomenon, and when we observe new data, we
update that knowledge to gain a more accurate understanding.

Bayes' Theorem is mathematically expressed as:

$$
P(\theta | D) = \frac{P(D | \theta) P(\theta)}{P(D)}.
$$

This equation can be interpreted as a rule for updating knowledge, where each component
represents a specific aspect of the learning process:

- **$P(\theta)$ – Prior Knowledge (Prior Distribution)**: Represents the initial beliefs
  about the model parameters before observing the data. For example, if we want to
  predict a person's height, the prior might state that most heights lie between 1.50 and
  2.00 meters, with an average around 1.70 meters.
- **$P(D | \theta)$ – Data Compatibility (Likelihood)**: Measures how likely the observed
  data are given a specific set of parameters. Continuing the previous example, if the
  model parameters suggest an average height of 1.80 meters, the likelihood evaluates how
  compatible the observed heights are with that prediction.
- **$P(D)$ – Normalization (Evidence)**: Acts as a normalization factor ensuring that the
  posterior distribution sums to one (since probabilities must lie between 0 and 1),
  satisfying the properties of a valid probability distribution. This term represents the
  total probability of observing the data under all possible parameter values.
- **$P(\theta | D)$ – Updated Knowledge (Posterior Distribution)**: This is the final
  result of the Bayesian process: the updated beliefs about the parameters after
  considering both the prior knowledge and the observed data. The posterior distribution
  combines prior information with empirical evidence to provide a more informed estimate
  of the parameters.

## Probabilistic Parameter Modeling in BNNs

In a BNN, each weight and bias is represented by a **probability distribution**,
typically a normal distribution with mean 0 and standard deviation 1, denoted as
$\mathcal{N}(0, 1)$. The training process does not aim to estimate a single value for
each parameter but rather to adjust the **posterior distribution** that best explains the
observed data.

This approach requires parameterizing the distributions through the mean and standard
deviation, updating them iteratively during training. The goal is to learn a **posterior
distribution $P(\theta | D)$** over the parameters $\theta$ given the data $D$, where:

- The **prior distribution** $P(\theta)$ typically assumes a standard Gaussian form,
  representing prior knowledge about the parameters.
- The **posterior distribution** $P(\theta | D)$ is adjusted during training and can
  differ significantly from the prior, shifting to reflect the knowledge gained from the
  data.

## Approximation Methods for the Posterior Distribution

Since exact computation of the posterior distribution is computationally intractable in
most practical cases, approximate inference techniques are employed:

- **Variational Inference**: Approximates the posterior distribution with a simpler
  distribution $q(\theta)$, optimizing the Kullback-Leibler (KL) divergence between
  $q(\theta)$ and $P(\theta | D)$. This method offers computational efficiency and
  scalability for large models, making it the most common choice in practical
  applications.

- **Markov Chain Monte Carlo (MCMC)**: Sampling-based methods that approximate the
  posterior by generating multiple samples. Although computationally more expensive, they
  provide more accurate approximations of the posterior and are useful when precision is
  prioritized over efficiency.

### ELBO Loss Function

Optimization in Bayesian Neural Networks is fundamentally based on maximizing the
Evidence Lower Bound (ELBO):

$$
\mathcal{L} = \mathbb{E}_{q(\theta)}[\log P(D | \theta)] - KL(q(\theta) || P(\theta)).
$$

This objective function balances two critical components that are essential for Bayesian
learning. The first component, known as the likelihood term
$\mathbb{E}_{q(\theta)}[\log P(D | \theta)]$, maximizes the probability of the observed
data under the approximate distribution $q(\theta)$. This component ensures that the
model maintains a good fit to the training data by encouraging the approximate posterior
to assign high probability to parameter values that explain the observed data well.

The second component, referred to as the regularization term
$KL(q(\theta) || P(\theta))$, minimizes the Kullback-Leibler divergence between the
approximate posterior distribution $q(\theta)$ and the prior distribution $P(\theta)$.
This component acts as a regularizing force that prevents overfitting by maintaining the
posterior distribution close to the prior when data is insufficient or ambiguous.

The KL divergence is formulated differently depending on the type of distribution. For
discrete distributions, the divergence is calculated as:

$$
KL(P || Q) = \sum_{x} P(x) \log \frac{P(x)}{Q(x)}.
$$

For continuous distributions, the divergence is expressed as an integral over the
parameter space:

$$
KL(P || Q) = \int_{-\infty}^{\infty} p(x) \log \frac{p(x)}{q(x)} dx.
$$

This duality in formulation allows the Bayesian framework to be applied in both discrete
and continuous spaces, providing flexibility in modeling different types of parametric
uncertainty. The continuous formulation is particularly relevant for BNNs, where the
parameters typically follow continuous distributions such as Gaussians, enabling the
framework to capture smooth variations in parameter uncertainty across the continuous
parameter space.

## Inference and Uncertainty Quantification

During the inference phase, a BNN generates predictions by repeatedly sampling from the
weight distribution. This process typically involves multiple independent inferences
(commonly between 50 and 1000 repetitions) for the same input, producing a set of
predictions that allows:

- Calculating the **mean** of the predictions as the final estimate.
- Determining the **variance** or standard deviation as a quantitative measure of the
  **associated uncertainty**.

This ability to quantify uncertainty is the main advantage of BNNs, providing insight
into the reliability of each individual prediction.

## Applications and Comparative Advantages

### Application Domains

BNNs are particularly valuable in contexts where uncertainty quantification is critical:

- **Biochemistry and drug discovery**: Risk and reliability assessment of new molecules.
- **Medical diagnosis**: Probabilistic estimation of critical diagnoses where uncertainty
  must be explicit.
- **Finance**: Risk assessment based on probabilistic predictions.
- **Robotics and reinforcement learning**: Adapting to dynamic environments under
  uncertainty.
- **Telecommunications**: Dynamic adjustment of network parameters considering
  environmental variability.

### Advantages over Deterministic Models

BNNs offer several advantages over traditional neural networks:

- **Formal uncertainty quantification**: Enables understanding of the model's limitations
  on new inputs, providing crucial information for decision-making in critical domains.
- **Effective regularization**: Prior distributions and KL divergence terms act as
  natural regularization mechanisms, significantly reducing the risk of overfitting.
- **Improved performance with limited data**: Prior knowledge acts as a guide when
  available data is scarce, improving the model's generalization.
- **Greater interpretability**: Facilitates analysis of prediction reliability and
  provides additional tools for informed decision-making, especially important in
  high-risk applications.

## Integration with Probabilistic Programming

BNNs naturally integrate with **probabilistic programming**, a paradigm that allows
complex statistical models to be described using declarative code. This integration
significantly broadens their applicability and facilitates implementation in systems
where explicit modeling of uncertainty is essential.

The combination provides a unified framework for developing applications that require
both the representational power of neural networks and the uncertainty modeling
capabilities of Bayesian inference.
