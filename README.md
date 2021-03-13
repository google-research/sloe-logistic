# Code to run experiments in *SLOE: A Faster Method for Statistical Inference in High-Dimensional Logistic Regression*.

Not an official Google product.

## Method Introduction
This library provides statistical inference for high dimensional logistic
regression maximum likelihood, based largely on the breakthrough results from
Sur and Candès (PNAS, 2019). The challenge with applying their results is that
they depend on an unobserved signal strength quantity. Our method estimates this
quantity a leave-one-out approach approach, which we outline in a forthcoming
paper.

By high-dimensions, we mean that the ratio of the number of covariates p to the
sample size `n` is strictly between 0 and 0.5. When the number of covariates is
too large, the data is separable, and our method will not help to recover from
such a case. When the number of covariates is small (say, `p <= 0.05 * n`), and
high dimensional adjustment is a bit numerically unstable, and adds little value
over the standard large-sample theory.

The setting studied is complementary to sparse high dimensional regimes. We
assume that there are a relatively large number of covariates that are weakly
correlated with the binary outcome. If one expects only a very small number of
the many candidate covariates to have a nonzero coefficient in the model,
sparse model selection and post-selective inference is probably a better
approach than the one taken here.

## Installation and tests
Run `run.sh` to install requirements and package, and run tests.

## Usage
The main approach proposed in our work is implemented in the
`UnbiasedLogisticRegression` class in `unbiased_logistic_regression.py`. This
has an `sklearn`-like interface, with a `fit`, `decision_function` and
`predict_proba` API. Additionally, for inference, we've added a
`prediction_intervals` method. See the inline documentation for more details
of usage.
