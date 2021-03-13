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

"""Tests for experiment_helpers."""

from absl.testing import absltest
from sloe_logistic.sloe_experiments import experiment_helpers


class ExperimentHelpersTest(absltest.TestCase):

  def test_simulation(self):
    params = experiment_helpers.SimulationParams(4000, 400, 1, seed=202103)
    sim = experiment_helpers.Simulation(params)
    features, outputs = sim.sample()

    self.assertAlmostEqual(features.mean(), 0, places=3)
    self.assertAlmostEqual(outputs.mean(), 0.5, places=2)

  def test_gwas_simulation(self):
    params = experiment_helpers.SimulationParams(4000, 400, 1, seed=202103)
    params.covariates = 'gwas'
    sim = experiment_helpers.GWASSimulation(params)
    features, outputs = sim.sample()

    self.assertAlmostEqual(features.mean(), 0, places=3)
    self.assertAlmostEqual(outputs.mean(), 0.5, places=2)

  def test_gwas_simulation_checks_covariates(self):
    params = experiment_helpers.SimulationParams(4000, 400, 1, seed=202103)
    params.covariates = 'not_gwas'
    with self.assertRaises(ValueError):
      _ = experiment_helpers.GWASSimulation(params)

if __name__ == '__main__':
  absltest.main()
