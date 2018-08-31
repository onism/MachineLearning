# Normalizing Flows Tutorial

ref: https://blog.evjang.com/2018/01/nf1.html

## Background

Statistical Machine Learning algorithms try to learn the structure of data by fitting a parameteric distribution $p(x;\theta)$ to it. Given a dataset, if we can represent it with a distribution, we can

1. Generate new data
2. Evaluate the likelihood of data observed at test time
3. Find the conditional relationship between variables
4. Score our algorithm by using complexity measures like entropy, MI, and moments of the distribution.

The Gaussian distribution is too simplistic. We should find a better distribution with the following properties:
1. Complex enough to model rich
2. while retaining the easy comforts of a Normal distribution

Some approaches can be used:
1. Mixture model
2. Autoregressive factorizations of policy/value distribution
3. Recurrent policies, noise, distribuional RL.
4. Learning with energy-based models
5. Normalizing Flows: **learn invertible, volume-tracking transormations of distributions that we can manipulate easily**

## Change of Variables, Change of Volume

Mathematically, locally-linear change in volume is $det(J(f^{-1}(x)))$, where $ J(f^{-1}(x))$ is the Jacobian of the function inverse - a higher dimensional generalization of the quantity $dx/dy$.

$$
y = f(x)
$$

$$
p(y) = p(f^{-1}(y)) \cdot |\text{det} J(f^{-1}(y))|
$$

$$
\log p(y) = \log p(f^{-1}(y)) + \log |\text{det}(J(f^{-1}(y)))|
$$

## Normalizing Flows

Any number of bijectors can be chained together, much like a neural network. This is construct is known as a  ``Normalizing Flow".

