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

"""Run experiment to understand convergence of SLOE estimator of eta.

Tests the SLOE estimator empirically by computing it
over a range of sample sizes for a bunch of different seeds, and storing in
csv files to be analyzed in a colab.
"""


from absl import app
from absl import flags
import apache_beam as beam
from apache_beam.options import pipeline_options
import numpy as np
import sloe_logistic.sloe_experiments.experiment_helpers as exp_helper
import statsmodels.api as sm

FLAGS = flags.FLAGS

flags.DEFINE_string('output_path', '/tmp/counts.txt', 'The output file path')
flags.DEFINE_integer('num_sims', 100, 'number of simulations to run')
flags.DEFINE_string('img_path', '/tmp/counts.png', 'Path to save plots')

N_RANGE = [250, 500, 1000, 2000, 4000]


def multiple_sample_sizes(seed):
  """Run same seed over multiple sample sizes."""
  for n in N_RANGE:
    yield [n, seed]


def run_sim(params):
  """Runs simulation and computes estimated eta_hat to compare to truth."""
  n = params[0]
  seed = params[1]
  kappa = FLAGS.features_per_sample
  p = int(n * kappa)

  gamma = np.sqrt(FLAGS.signal_strength)
  rand_state = np.random.RandomState(201216 + seed)

  p_positive = int(p / 8)
  p_negative = p_positive
  p_zero = p - p_positive - p_negative
  beta = 2 * np.concatenate(
      (np.ones(p_positive), -np.ones(p_negative), np.zeros(p_zero)))
  beta *= gamma

  features = rand_state.randn(n, p) / np.sqrt(p)
  labels = (rand_state.rand(n) <= 1.0 /
            (1.0 + np.exp(-features.dot(beta)))).astype(float)

  logit_model = sm.Logit(labels, features)
  logit_model_fit = logit_model.fit(disp=False)
  beta_hat = logit_model_fit.params

  hessian = logit_model.hessian(beta_hat)
  # Computes X_i^T H^{-1} X_i for all examples. Used in Sherman-Morrison formula
  # below.
  xi_hessian_inv_xi = np.diag(
      features.dot(np.linalg.solve(hessian, features.T)))
  pred = logit_model_fit.predict(features)
  # Sherman-Morrison formula for X_i^T H_{-i}^{-1} X_i, where H_{-i} is Hessian
  # without i-th example.
  mod = xi_hessian_inv_xi / (1.0 + xi_hessian_inv_xi * pred * (1 - pred))
  infl = mod * (labels - pred) + features.dot(beta_hat)

  eta_hat = np.var(infl)

  eta_hat_simp = np.linalg.norm(beta_hat)**2

  return np.array([n, seed, eta_hat, eta_hat_simp])


def main(unused_argv):
  # If you have custom beam options add them here.
  beam_options = pipeline_options.PipelineOptions()

  with beam.Pipeline(options=beam_options) as pipe:
    _ = (
        pipe
        | beam.Create(range(FLAGS.num_sims))
        | beam.FlatMap(multiple_sample_sizes)
        | 'PrepShuffle' >> beam.Reshuffle()
        | beam.Map(run_sim)
        | beam.Map(exp_helper.numpy_array_to_csv)
        | beam.Reshuffle()
        |
        'WriteToText' >> beam.io.WriteToText(FLAGS.output_path, num_shards=5))


if __name__ == '__main__':
  app.run(main)
