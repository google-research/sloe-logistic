# coding=utf-8
# Copyright 2021 The SLOE Logistic Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Implements logistic regression w/ ProbeFrontier estimator of bias correction.

Implements the bias correction and inference for the MLE using the ProbeFrontier
estimator of the signal strength as in [1]. Theory for arbitrary covariance with
Gaussian features from [2], and empirical evidence suggesting good performance
for non-Gaussian designs.

[1] Sur, Pragya, and Emmanuel J. Candès. "A modern maximum-likelihood theory
for high-dimensional logistic regression." Proceedings of the National Academy
of Sciences 116.29 (2019): 14516-14525.
[2] Zhao, Qian, Pragya Sur, and Emmanuel J. Candes. "The asymptotic distribution
of the mle in high-dimensional logistic models: Arbitrary covariance." arXiv
preprint arXiv:2001.09351 (2020).
"""
from absl import app
import numpy as np
import scipy
from sloe_logistic import asymp_system_solve
from sloe_logistic import unbiased_logistic_regression
import statsmodels.api as sm
import statsmodels.tools


class ProbeFrontierLogisticRegression(
    unbiased_logistic_regression.UnbiasedLogisticRegression):
  """Implements ProbeFrontier and statistical inference with it."""

  def __init__(self, num_subsamples=10):
    super().__init__(fit_intercept=False)
    self.num_subsamples = num_subsamples
    self.sep_calls = 0

  def fit(self, features, outcome, weights=None, verbose=False):
    """Fit ProbeFrontier model."""
    if self.fit_intercept:
      raise NotImplementedError("ProbeFrontier doesn't work with intercept")
    self.sep_calls = 0

    self.sm.fit(features, outcome, weights)

    if weights is None:
      weights = 1

    kappa = float(features.shape[1]) / features.shape[0]
    gamma_hat = self.estimate_gamma(features, outcome)

    self.alpha, _, sigma, _ = asymp_system_solve.correction_factors(
        kappa, None, gamma_hat, 0, use_eta=False)

    self.coef_ = self.sm.coef_ / self.alpha
    self.intercept_ = 0

    self._set_coef_cov(features, sigma / np.sqrt(kappa), self.alpha)

    return self, self.sep_calls

  def estimate_gamma(self, features, outcome):
    """Estimate gamma."""
    estimated_kappa_threshold = self.probe_frontier(features, outcome)
    if estimated_kappa_threshold < 0:
      print(features, outcome)
    if estimated_kappa_threshold >= 0.499:
      return 0.0
    return asymp_system_solve.frontier(estimated_kappa_threshold)

  def probe_frontier(self, features, outcome):
    """Probe for frontier."""
    n, p = features.shape
    upper_frac = n
    lower_frac = min(n, 1.99 * p)
    obs = []
    while abs(upper_frac - lower_frac) > (0.05 * p):
      frac = int((upper_frac + lower_frac) / 2)
      p_sep = 0
      for _ in range(self.num_subsamples):
        indices = np.random.choice(n, frac, replace=False)
        feature_sub = features[indices, :]
        outcome_sub = outcome[indices]
        p_sep += self.is_separable(feature_sub, outcome_sub)
      p_sep /= float(self.num_subsamples)
      obs.append([frac, p_sep])
      if p_sep >= 0.8:
        lower_frac = frac
      elif p_sep <= 0.2:
        upper_frac = frac
      elif p_sep > 0.5:
        lower_frac = 0.5 * lower_frac + 0.5 * frac
      else:
        upper_frac = 0.5 * upper_frac + 0.5 * frac

    if len(obs) <= 2:
      frac = int(0.5 * (upper_frac + lower_frac))
    else:
      obs = np.array(obs)

      if (obs[0, 1] > (1 - 1.5 / self.num_subsamples)):
        frac = obs[0, 0]
      elif (obs[-1, 1] < (1.5 / self.num_subsamples)):
        frac = obs[-1, 0]
      else:
        try:
          interp = sm.GLM(
              obs[:, 1],
              sm.add_constant(obs[:, 0].reshape(-1, 1)),
              family=sm.families.Binomial())
          res = interp.fit()
          frac = -res.params[0] / res.params[1]

        except statsmodels.tools.sm_exceptions.PerfectSeparationError:
          threshold = np.argmax(np.diff(obs[:, 1], prepend=0))
          frac = obs[threshold, 0]

    return min(float(p) / frac, 0.5)

  def is_separable(self, features, outcome):
    """Check whether data are linearly separable."""
    self.sep_calls += 1
    n, p = features.shape
    features_aug = np.ones((n, p + 1))
    features_aug[:, :-1] = features
    features_aug *= (2 * outcome - 1).reshape(-1, 1)
    b = -np.ones(n)
    res = scipy.optimize.linprog(
        b, A_eq=features_aug.T, b_eq=np.zeros(p + 1), method='interior-point')
    if res.status == 0:
      return res.fun > -1e-6
    elif res.status == 2:
      return False
    elif res.status == 3:
      return False
    else:
      print(res)
      raise Exception('Error finding separability')


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  p = ProbeFrontierLogisticRegression()

  features = np.random.randn(600, 300) / np.sqrt(300)
  outcome = (np.random.rand(600) <= 1 /
             (1.0 + np.exp(-1 * features.sum(axis=1)))).astype(float)
  primal = p.is_separable(features, outcome)
  print(primal)

  features = np.array([[1, 1], [0, 0]])
  outcome = np.array([1, 0])
  print(p.is_separable(features, outcome))
  features = np.array([[1, 1], [0, 0], [-1, -1]])
  outcome = np.array([1, 0, 1])
  print(p.is_separable(features, outcome))

  features = np.random.randn(100, 100)
  outcome = (np.random.rand(100) <= 0.5).astype(float)
  print(p.is_separable(features, outcome))

  features = np.random.randn(100, 10)
  outcome = (np.random.rand(100) <= 0.5).astype(float)
  print(p.is_separable(features, outcome))

if __name__ == '__main__':
  app.run(main)
