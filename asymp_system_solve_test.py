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
from sloe_logistic import asymp_system_solve


class AsympSystemSolveTest(absltest.TestCase):

  def test_correction_factors_solve(self):
    sol = asymp_system_solve.correction_factors(
        0.2, 1, np.sqrt(5), 0, use_eta=False)
    target = [1.499, 3.027, 2.1214, 0.0]
    for i in range(4):
      self.assertAlmostEqual(sol[i], target[i], places=3)

    sol = asymp_system_solve.correction_factors(
        0.1, 8.881028475794636, np.sqrt(5), 0, use_eta=True)
    target = [1.174, 1.007, 1.086, 0.0]
    for i in range(4):
      self.assertAlmostEqual(sol[i], target[i], places=3)

  def test_frontier(self):
    sol = asymp_system_solve.frontier(0.1)
    self.assertAlmostEqual(sol, 9.890, places=3)

    sol = asymp_system_solve.frontier(0.2)
    self.assertAlmostEqual(sol, 4.550, places=3)

if __name__ == '__main__':
  absltest.main()
