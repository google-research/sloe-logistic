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

"""Implements methods for inference for logistic regression based on the MLE.

Implements SLOE and other methods for inference for logistic regression
based on the MLE.
"""

import numpy as np
import scipy
import scipy.stats
import sklearn.linear_model

from sloe_logistic import asymp_system_solve


class ScaledLogisticRegression(object):
  """Generic class for methods rescaling the logistic regression MLE."""

  def __init__(self):
    pass

  def predict_proba(self, features, *args, **kwargs):
    del args
    del kwargs
    results = np.zeros((features.shape[0], 2))
    log_odds_ratio = features.dot(self.coef_.T).reshape(-1) + self.intercept_
    results[:, 1] = self._expit(log_odds_ratio)
    results[:, 0] = 1 - results[:, 1]
    return results

  def predict_inv_proba(self, features, *args, **kwargs):
    """Provides reciprocal of probability given features."""
    return 1 / self.predict_proba(features, *args, **kwargs)

  def _expit(self, logit, trimmed=False):
    if trimmed:
      logit = np.minimum(logit, 5)
      logit = np.maximum(logit, -5)
    return 1.0 / (1.0 + np.exp(-logit))


class PlattScaledLogisticRegression(ScaledLogisticRegression):
  """Rescales the logit reg MLE to make it calibrated using approximation."""

  def __init__(self, fit_intercept=True, **kwargs):
    del kwargs
    super().__init__()
    self.fit_intercept = fit_intercept
    self.sm = sklearn.linear_model.LogisticRegression(
        fit_intercept=fit_intercept,
        penalty="none",
        solver="newton-cg",
        warm_start=False)

  def fit(self, features, outcome, weights=None, verbose=False):
    """Compute MLE and then use Taylor approximation rescale for calibration."""
    del verbose
    self.sm.fit(features, outcome, weights)

    refit_weights = None
    if refit_weights is None:
      refit_weights = 1

    # Get leave-one-out logits to pass in to Platt scaling
    pred = self.sm.predict_proba(features)[:, 1]
    hessian = -features.T.dot(
        (refit_weights * pred * (1 - pred)).reshape(-1, 1) * features)
    xihinvxi = np.diag(features.dot(np.linalg.solve(hessian, features.T)))
    mod = xihinvxi / (1.0 + xihinvxi * refit_weights * pred * (1 - pred))
    features = mod * refit_weights * (
        outcome - pred) + self.sm.decision_function(features)

    # Fit model for outcome using LOO logit estimates as feature. Coefficient on
    # feature is scaling to recalibrate model.
    cm = sklearn.linear_model.LogisticRegression(
        penalty="none", fit_intercept=self.fit_intercept)
    cm.fit(features.reshape(-1, 1), outcome.reshape(-1), weights)
    self.coef_ = self.sm.coef_ * cm.coef_
    if self.fit_intercept:
      self.intercept_ = cm.coef_ * self.sm.intercept_ + cm.intercept_
    else:
      self.intercept_ = 0
    return self


class CVRegLogisticRegression(ScaledLogisticRegression):
  """Cross-validated regularized logistic regression MLE."""

  def __init__(self, fit_intercept=True, Cs=10, **kwargs):
    super().__init__(**kwargs)
    self.fit_intercept = fit_intercept
    self.sm = sklearn.linear_model.LogisticRegressionCV(
        fit_intercept=fit_intercept,
        Cs=Cs,
        penalty="l2",
        solver="newton-cg")

  def fit(self, features, outcome, weights=None, verbose=False):
    """Fit cross-validated model."""
    del verbose

    self.sm.fit(features, outcome, weights)

    if self.fit_intercept:
      self.intercept_ = self.sm.intercept_
    else:
      self.intercept_ = 0
    self.coef_ = self.sm.coef_

    return self


class LogisticRegressionInference(ScaledLogisticRegression):
  """Base class inference with logit reg that computes P/CIs from covariance."""

  def __init__(self, fit_intercept=True, ci=50, **kwargs):
    super().__init__(**kwargs)
    self.fit_intercept = fit_intercept
    self.coef_cov = None
    self.hessian = None
    self.chi_sq_rescale = 1
    self.set_coverage(ci)

  def set_coverage(self, ci):
    """Sets expected coverage level."""
    self.ci_coverage = ci / 100.0
    self.z = scipy.stats.norm.ppf(0.5 + self.ci_coverage / 2.0)

  def _set_coef_cov(self, *args):
    pass

  def _get_prediction_variances(self, features):
    if self.fit_intercept:
      features_aug = np.ones((features.shape[0], features.shape[1] + 1))
      features_aug[:, :-1] = features
    else:
      features_aug = features
    return (features_aug.dot(self.coef_cov) *
            features_aug).sum(axis=-1).reshape(-1)

  def p_values(self):
    """Get p-values for a fitted model using Wald test."""
    scale = np.sqrt(np.diag(self.coef_cov))
    if self.fit_intercept:
      scale = scale[:-1]
    t = np.abs(self.coef_) / scale
    t = t.reshape(-1)
    p = 2 * scipy.stats.norm.sf(t)
    return p

  def decision_function(self, features):
    """Compute logits (ie decision function in sklearn parlance."""
    return features.dot(self.coef_.T).reshape(-1) + self.intercept_

  def prediction_intervals(self, features, logit=False):
    """Computes prediction CI for each row of features using coef covariance."""
    if self.coef_cov is None:
      raise Exception(
          "No covariance matrix defined yet, so can't do inference.")

    logits = self.decision_function(features)
    variances = self._get_prediction_variances(features)

    lower_ci = logits - self.z * np.sqrt(variances)
    upper_ci = logits + self.z * np.sqrt(variances)

    results = np.zeros((features.shape[0], 3))
    results[:, 0] = lower_ci
    results[:, 1] = logits
    results[:, 2] = upper_ci

    if not logit:
      results = self._expit(results)

    return results

  def predict_proba(self, X):
    logits = self.decision_function(X)

    preds = self._expit(logits)

    results = np.zeros((X.shape[0], 2))
    results[:, 1] = preds
    results[:, 0] = 1 - preds
    return results

  def predict_inv_proba(self, X):
    logits = self.decision_function(X)

    pos_exps = np.exp(logits)
    neg_exps = np.exp(-logits)

    results = np.zeros((X.shape[0], 2))
    results[:, 1] = 1 + neg_exps
    results[:, 0] = 1 + pos_exps

    return results


class LogisticRegressionMLE(LogisticRegressionInference):
  """Computes the un-rescaled MLE and standard large-sample stats inference."""

  def __init__(self, fit_intercept=True, **kwargs):
    super().__init__(fit_intercept=fit_intercept, **kwargs)
    self.fit_intercept = fit_intercept
    self.sm = sklearn.linear_model.LogisticRegression(
        fit_intercept=fit_intercept,
        penalty="none",
        solver="newton-cg",
        warm_start=False)

  def fit(self, features, outcomes, weights=None, verbose=False):
    """Fit standard MLE model and compute coefficient covariance matrix."""
    del verbose

    self.sm.fit(features, outcomes, weights)

    self.coef_ = self.sm.coef_
    if self.fit_intercept:
      self.intercept_ = self.sm.intercept_
    else:
      self.intercept_ = 0

    self._set_coef_cov(features, weights)

    return self

  def _set_coef_cov(self, features, weights):
    """Use large-sample asymp. to compute coefficient covariance matrix."""
    if weights is None:
      weights = 1
    pred = self.sm.predict_proba(features)[:, 1]
    _, p = features.shape
    if self.fit_intercept:
      features_aug = np.ones((features.shape[0], features.shape[1] + 1))
      features_aug[:, :-1] = features
      dim = p + 1
    else:
      features_aug = features
      dim = p
    hessian = features_aug.T.dot(
        (weights * pred *
         (1 - pred)).reshape(-1, 1) * features_aug) / np.mean(weights)
    self.hessian = -hessian
    self.coef_cov = scipy.linalg.solve(hessian, np.eye(dim), assume_a="pos")


class LogisticRegressionPercBoot(LogisticRegressionInference):
  """Fit standard MLE using multiplier bootstrap and compute percentile CIs.

  It is not recommended to use this method in practice if d / n ~> 0.05. The
  results from our paper suggest that it is very biased and has poor precision.
  """

  def __init__(self, fit_intercept=True, num_boot=20, **kwargs):
    super().__init__(fit_intercept=fit_intercept, **kwargs)
    self.fit_intercept = fit_intercept
    self.sm = sklearn.linear_model.LogisticRegression(
        fit_intercept=fit_intercept,
        penalty="none",
        solver="newton-cg",
        warm_start=False)
    self.num_boot = num_boot

  def fit(self, features, outcome, weights=None, verbose=False):
    """Fit main model and bootstrapped models with multiplier bootstrap."""
    del verbose
    self.sm.fit(features, outcome, weights)

    self.coef_ = self.sm.coef_
    if self.fit_intercept:
      self.intercept_ = self.sm.intercept_
    else:
      self.intercept_ = 0

    if weights is None:
      weights = 1.0

    n = features.shape[0]
    self.bootstraps = []
    for _ in range(self.num_boot):
      self.sm.fit(features, outcome,
                  weights * np.random.poisson(lam=1.0, size=n))
      if np.linalg.norm(self.sm.coef_) >= 1e6:
        continue
      d = {"coef": self.sm.coef_.reshape(-1)}
      if self.fit_intercept:
        d["intercept"] = self.sm.intercept_
      else:
        d["intercept"] = 0
      self.bootstraps.append(d)

    return self

  def p_values(self):
    raise NotImplementedError(
        "This form of bootstrap does not lend itself well to p-values")

  def approx_lrt_p_values(self):
    raise NotImplementedError(
        "This form of bootstrap does not lend itself well to p-values")

  def _predict_with_param_dict(self, params, features):
    return features.dot(params["coef"]).reshape(-1) + params["intercept"]

  def prediction_intervals(self, X, logit=False):
    """Computes percentile CIs for feature rows using bootstrap samples."""
    all_preds = np.array(
        [self._predict_with_param_dict(d, X) for d in self.bootstraps])

    ci_range = (1 - self.ci_coverage) / 2
    results = np.quantile(all_preds, q=(ci_range, 0.5, 1 - ci_range), axis=0).T

    if not logit:
      results = self._expit(results)

    return results


class UnbiasedLogisticRegression(LogisticRegressionInference):
  """Corrected bias and inference with the logitistic regression MLE."""

  def __init__(self, fit_intercept=False, **kwargs):
    super().__init__(fit_intercept, **kwargs)

    self.fit_intercept = fit_intercept
    if fit_intercept:
      raise ValueError("This model doesn't allow fitting an intercept.")

    self.sm = sklearn.linear_model.LogisticRegression(
        fit_intercept=fit_intercept,
        penalty="none",
        solver="newton-cg",
        warm_start=False)

  def fit(self, features, outcome, weights=None, verbose=False):
    """Fit MLE, estimate eta with SLOE, de-bias, and estimate covariance."""
    del verbose
    kappa = float(features.shape[1]) / features.shape[0]

    self.sm.fit(features, outcome, weights)

    if weights is None:
      weights = 1

    pred = self.sm.predict_proba(features)[:, 1]
    weights /= np.mean(weights)
    diag = weights * pred * (1 - pred)
    hessian = -features.T.dot(diag.reshape(-1, 1) * features)
    self.hessian = hessian
    xihinvxi = np.einsum("ij,ji->i", features,
                         np.linalg.solve(hessian, features.T))
    mod = xihinvxi / (1.0 + xihinvxi * diag)
    infl = mod * weights * (outcome -
                            pred) + self.sm.decision_function(features)

    eta_hat = np.var(infl)

    b0 = 0

    self.alpha, lambda_, sigma, intercept_est = asymp_system_solve.correction_factors(
        kappa, eta_hat, np.sqrt(eta_hat), b0, use_eta=True)
    if (kappa >= 0.05 and self.alpha < 0.999) or self.alpha > 5 \
        or lambda_ < 0.1 or sigma < 0.3 or lambda_ > 1e3 or sigma > 1e3:
      raise ValueError("Problem with optimization")

    self.eta_hat = eta_hat
    self.lambda_ = lambda_
    self.sigma = sigma
    self.intercept_est = intercept_est

    self.chi_sq_rescale = lambda_ * self.alpha**2 / sigma**2
    self.coef_ = self.sm.coef_ / self.alpha
    self.intercept_ = 0

    self._set_coef_cov(features, sigma / np.sqrt(kappa), self.alpha)

    return self

  def _set_coef_cov(self, features, sigma, alpha):
    n, p = features.shape
    features_aug = features
    dim = p
    feature_cov = features_aug.T.dot(features_aug)
    one_on_tau_sq = scipy.linalg.solve(feature_cov, np.eye(dim), assume_a="pos")
    self.coef_cov = one_on_tau_sq
    self.coef_cov *= (1 - float(p) / n) * ((sigma / alpha)**2)
