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

"""Tests for sloe_logistic.asymp_system_solve."""

from absl.testing import absltest
import numpy as np
from sloe_logistic import unbiased_logistic_regression


class UnbiasedLogisticRegressionTest(absltest.TestCase):

  def get_simulated_data(self, n, d):
    np.random.seed(1)
    features = np.random.randn(n, d)
    beta = np.sqrt(5 * 2.0 / d) * np.ones(d)
    beta[(d // 2):] = 0

    outcome = (np.random.rand(n) <= 1.0 /
               (1.0 + np.exp(-features.dot(beta)))).astype(float)

    return features, outcome

  def test_unbiased_model(self):
    """Tests that UnbiasedLogisticRegression.fit runs without errors."""
    n, d = 1000, 100
    features, outcome = self.get_simulated_data(n, d)
    model = unbiased_logistic_regression.UnbiasedLogisticRegression(
        fit_intercept=False)
    model.fit(features, outcome)

    self.assertLen(model.coef_.reshape(-1), features.shape[1])

  def test_cant_fit_intercept(self):
    """Tests that UnbiasedLogisticRegression doesn't allow fit_intercept.

    Currently, there's no support for fitting the intercept. This checks that
    trying to fit an intercept raises an error instead of silently ignoring
    the intercept.
    """
    with self.assertRaises(ValueError):
      _ = unbiased_logistic_regression.UnbiasedLogisticRegression(
          fit_intercept=True)

  def test_platt_model(self):
    """Tests that PlattScaledLogisticRegression.fit runs without errors."""
    n, d = 1000, 100
    features, outcome = self.get_simulated_data(n, d)
    model = unbiased_logistic_regression.PlattScaledLogisticRegression(
        fit_intercept=False)
    model.fit(features, outcome)

  def test_standard_mle_model(self):
    """Tests that LogisticRegressionMLE.fit runs without errors."""
    n, d = 1000, 100
    features, outcome = self.get_simulated_data(n, d)
    model = unbiased_logistic_regression.LogisticRegressionMLE(
        fit_intercept=False)
    model.fit(features, outcome)

  def test_bootstrap_model(self):
    """Tests that LogisticRegressionPercBoot.fit runs without errors."""
    n, d = 1000, 100
    features, outcome = self.get_simulated_data(n, d)
    model = unbiased_logistic_regression.LogisticRegressionPercBoot(
        fit_intercept=False)
    model.fit(features, outcome)

  def test_bootstrap_prediction_intervals(self):
    """Tests that LogisticRegressionPercBoot.prediction_intervals runs."""
    n, d = 1000, 100
    features, outcome = self.get_simulated_data(n, d)
    model = unbiased_logistic_regression.LogisticRegressionPercBoot(
        fit_intercept=False)
    model.fit(features, outcome)
    model.prediction_intervals(features)

  def test_regularized_model(self):
    """Tests that CVRegLogisticRegression.fit runs without errors."""
    n, d = 1000, 100
    features, outcome = self.get_simulated_data(n, d)
    model = unbiased_logistic_regression.CVRegLogisticRegression(
        fit_intercept=False)
    model.fit(features, outcome)

    self.assertLen(model.coef_.reshape(-1), features.shape[1])

  def test_prediction_intervals(self):
    n, d = 1000, 100
    features, outcome = self.get_simulated_data(n, d)
    model = unbiased_logistic_regression.UnbiasedLogisticRegression(
        fit_intercept=False)
    model.fit(features, outcome)

    test_features, _ = self.get_simulated_data(100, d)
    intervals = model.prediction_intervals(test_features)
    estimated_probs = model.predict_proba(test_features)[:, 1]

    np.testing.assert_array_less(intervals[:, 0], estimated_probs)
    np.testing.assert_array_less(estimated_probs, intervals[:, 2])

  def test_corrected_p_values(self):
    """Check null P value CDF is within 95% CI of uniform CDF."""
    n, d = 4000, 400
    features, outcome = self.get_simulated_data(n, d)
    model = unbiased_logistic_regression.UnbiasedLogisticRegression(
        fit_intercept=False)
    model.fit(features, outcome)

    thresh = 0.1
    emp_p_cdf = model.p_values().reshape(-1)[(d // 2):] <= thresh
    self.assertAlmostEqual(
        emp_p_cdf.mean(),
        thresh,
        delta=1.96 * emp_p_cdf.std() / np.sqrt(d // 2))


if __name__ == '__main__':
  absltest.main()
