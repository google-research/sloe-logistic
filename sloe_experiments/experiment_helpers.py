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

"""Helpers used across many experiments to understand SLOE estimator.

Implements the simulation settings studied in the paper
and provides a bunch of helper functions used throughout to create and analyze
simulations.
"""


from absl import flags
import numpy as np

from sloe_logistic import probe_frontier
from sloe_logistic import unbiased_logistic_regression

FLAGS = flags.FLAGS

flags.DEFINE_enum(
    "covariates", "gaussian", ["gaussian", "gwas"],
    "Covariate generating distribution for sim. If gaussian, see --covariance"
    "for more details about distribution.")
flags.DEFINE_enum(
    "covariance", "isotropic", ["isotropic", "elliptical"],
    "Covariance of covariates.")
flags.DEFINE_float("features_per_sample", 0.2,
                   "number of features per sample (kappa)")
flags.DEFINE_float("intercept", 0, "intercept of logits")
flags.DEFINE_enum(
    "method", "newmethod", ["newmethod", "mle", "probefrontier"],
    "Which method for estimation and inference?")
flags.DEFINE_boolean("one_and_none", False,
                     "Put all of the signal in one (the first) covariate. "
                     "This does not meet assumptions of method, but provides "
                     "a nice robustness check to see how inaccurate results "
                     "will be.")
flags.DEFINE_integer("sample_size", 1000, "number of samples per simulation")
flags.DEFINE_float("signal_strength", 5, "variance of logits (gamma^2)")
flags.DEFINE_boolean(
    "uncentered", False,
    "By default, covariates are centered. This makes them uncentered (w/o effecting intercept)?."
)


class SimulationParams(object):
  """Simulation parameters shared across SLOE estimator experiments."""

  def __init__(self,
               training_n,
               p,
               gamma,
               covariates="gaussian",
               covariance="isotropic",
               one_and_none=False,
               uncentered=False,
               intercept=0,
               seed=None):
    self.training_n = training_n
    self.p = p
    self.gamma = gamma
    self.covariates = covariates
    self.covariance = covariance
    self.one_and_none = one_and_none
    self.uncentered = uncentered
    self.intercept = intercept
    self.seed = seed

  @classmethod
  def create_from_flags(cls):
    """Create a SimulationParams object from FLAGS."""
    n = FLAGS.sample_size
    kappa = FLAGS.features_per_sample
    gamma = np.sqrt(FLAGS.signal_strength)
    covariates = FLAGS.covariates
    covariance = FLAGS.covariance
    one_and_none = FLAGS.one_and_none
    uncentered = FLAGS.uncentered
    intercept = FLAGS.intercept

    p = int(n * kappa)
    return SimulationParams(n, p, gamma, covariates, covariance, one_and_none,
                            uncentered, intercept)


class Simulation(object):
  """Standard simulation model used in most experiments in SLOE paper."""

  def __init__(self, simulation_params):
    self.simulation_params = simulation_params

    self._check_sim_params()
    self._reset_random_state()
    self._initialize_params()

  def _initialize_params(self):
    """Initializes statistical params of model from simulation parameters."""
    p = self.simulation_params.p

    self.intercept_ = self.simulation_params.intercept

    if self.simulation_params.one_and_none:
      self.beta = np.zeros(p)
      self.beta[0] = self.simulation_params.gamma * np.sqrt(p)
    else:
      self.p_positive = int(p / 8)
      self.p_negative = self.p_positive
      self.p_zero = p - self.p_positive - self.p_negative
      self.beta = 2 * np.concatenate((np.ones(
          self.p_positive), -np.ones(self.p_negative), np.zeros(self.p_zero)))
      self.beta *= self.simulation_params.gamma

    if self.simulation_params.covariance == "isotropic":
      self.diag = np.ones(p)
    elif self.simulation_params.covariance == "elliptical":
      self.diag = self.random_state.rand(p) + 0.5
      self.diag /= self.diag[:(self.p_positive + self.p_negative)].mean()
      self.diag[0] = 1
    else:
      raise NotImplementedError("No covariance {}".format(
          self.simulation_params.covariance))

    if self.simulation_params.uncentered:
      self.centering = np.ones(p)
      self.intercept_ -= self.beta.dot(self.centering)
    else:
      self.centering = 0

  def null_indices(self):
    """Get null indices."""
    return slice(-self.p_zero, None, None)

  def _check_sim_params(self):
    if self.simulation_params.covariates != "gaussian":
      raise ValueError(
          "Simulation parameters calls for {} covariate distribution, "
          "but this class generates Gaussian covariates.".format(
              self.simulation_params.covariates))

  def _reset_random_state(self):
    self.random_state = np.random.RandomState(seed=self.simulation_params.seed)

  def _sample_x(self, n):
    return self.diag * self.random_state.randn(
        n, self.simulation_params.p) / np.sqrt(
            self.simulation_params.p) + self.centering

  def sample(self, n=None):
    """Sample data from simulation."""
    if n is None:
      n = self.simulation_params.training_n

    x1 = self._sample_x(n)
    y1 = (self.random_state.rand(n) <= 1.0 /
          (1.0 + np.exp(-x1.dot(self.beta) - self.intercept_))).astype(float)
    return (x1, y1)


class GWASSimulation(Simulation):
  """From Sur and Candes, 2019. PNAS. Section 4(g)."""

  def __init__(self, simulation_params):
    super().__init__(simulation_params)

    self._initialize_cov_params()

  def _initialize_cov_params(self):
    self.equil = 0.5 * self.random_state.rand(self.simulation_params.p) + 0.25

  def _check_sim_params(self):
    if self.simulation_params.covariates != "gwas":
      raise ValueError(
          "Simulation parameters calls for {} covariate distribution, "
          "but this class generates GWAS-like covariates.".format(
              self.simulation_params.covariates))

  def covariate_mean(self):
    return 2 * (1 - self.equil)

  def covariate_std(self):
    return 2 * (1 - self.equil) * self.equil

  def _sample_x(self, n):
    p = self.simulation_params.p
    x1 = np.zeros((n, p))
    equil = self.equil
    for j in range(p):
      pj = equil[j]
      probs = np.array([pj**2, 2 * pj * (1 - pj), (1 - pj)**2])
      x1[:, j] = self.random_state.choice(3, size=(n,), p=probs)
    x1 -= self.covariate_mean().reshape(1, -1)
    x1 /= self.covariate_std().reshape(1, -1) * np.sqrt(p)
    return x1


def multiple_sim_params(seed, kappa_range, gamma_range):
  """For each seed, map to a variety of simulation parameters."""
  for kappa in kappa_range:
    for gamma in gamma_range:
      yield [kappa, gamma, seed]


def multiple_sample_sizes(seed, n_range):
  """For each seed, map to a variety of sample sizes."""
  for n in n_range:
    yield [n, seed]


def create_sim(sim_params):
  """Create a simulation according to passed params."""
  if sim_params.covariates == "gaussian":
    return Simulation(sim_params)
  elif sim_params.covariates == "gwas":
    return GWASSimulation(sim_params)
  else:
    raise NotImplementedError("No simulation with covariates {}".format(
        FLAGS.covariates))


def create_inference_model(method=None, fit_intercept=False):
  """Create a model to use for inference, getting default from FLAGS."""
  if method is None:
    method = FLAGS.method

  if method == "probe_frontier":
    if fit_intercept:
      raise NotImplementedError(
          "ProbeFrontier can't fit an intercept right now")
    logit_model = probe_frontier.ProbeFrontierLogisticRegression(
        num_subsamples=8)
  elif method == "mle":
    logit_model = unbiased_logistic_regression.LogisticRegressionMLE(
        fit_intercept=fit_intercept)
  elif method == "bootstrap":
    logit_model = unbiased_logistic_regression.LogisticRegressionPercBoot(
        fit_intercept=fit_intercept)
  elif method == "newmethod":
    logit_model = unbiased_logistic_regression.UnbiasedLogisticRegression(
        fit_intercept=fit_intercept)
  else:
    raise NotImplementedError("No method {}".format(FLAGS.method))
  return logit_model


def numpy_array_to_csv(arr):
  return ",".join(["%.5f" % num for num in arr])


