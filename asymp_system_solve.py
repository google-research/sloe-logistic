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

"""Solves nonlinear equations for high dim correction factors for MLE.

Solves the nonlinear equations in Sur and Candès (PNAS., 2019) to find
the adjustment factors for bias and variance of logistic regression MLE.
"""


import functools

from absl import app
import numpy as np
import scipy
import scipy.integrate
import scipy.optimize

import sloe_logistic.mle_param_integrands as mle_helper


def _t_integrand(z, v, t, gamma):
  """Integrand used to calculate when the logistic MLE exists."""
  return 2 * mle_helper.sigmoid(gamma * v) * mle_helper.pdf(z, v) * (
      max(z - t * v, 0)**2)


def _t_problem(t, gamma):
  """Minimizer of this integrand in t is the frontier where the MLE exists."""
  loss, _ = scipy.integrate.dblquad(_t_integrand, -8, 8, -8, 8, (
      t,
      gamma,
  ), 1e-6, 1e-6)
  return loss


def _g_mle_inv(gamma):
  """Frontier where data separable in limit. Gives kappa in terms of gamma."""
  res = scipy.optimize.minimize_scalar(
      _t_problem, bounds=(-10, 10), args=(gamma,), method='Bounded')
  return _t_problem(res.x, gamma)


def frontier(kappa):
  """Frontier where data separable in limit. Gives gamma in terms of kappa."""
  gamma_star = scipy.optimize.brentq(lambda gamma: _g_mle_inv(gamma) - kappa, 0,
                                     25)
  return gamma_star


def equations(kappa, eta, gamma, beta0, use_eta, alpha, lambda_, sigma, b0):
  """The solution to these equations gives the high dimensional adjustment."""
  if use_eta:
    gamma = np.sqrt(max(eta - sigma**2, 0.0001))
  else:
    gamma *= alpha

  eq1, _ = scipy.integrate.dblquad(
      mle_helper.integrand, -8, 8, -8, 8,
      (kappa, gamma, b0, alpha, lambda_, sigma, beta0, 1), 1e-4, 1e-4)
  eq2, _ = scipy.integrate.dblquad(
      mle_helper.integrand, -8, 8, -8, 8,
      (kappa, gamma, b0, alpha, lambda_, sigma, beta0, 2), 1e-4, 1e-4)
  eq3, _ = scipy.integrate.dblquad(
      mle_helper.integrand, -8, 8, -8, 8,
      (kappa, gamma, b0, alpha, lambda_, sigma, beta0, 3), 1e-4, 1e-4)
  eq4, _ = scipy.integrate.dblquad(
      mle_helper.integrand, -8, 8, -8, 8,
      (kappa, gamma, b0, alpha, lambda_, sigma, beta0, 4), 1, 1)
  eq1 -= sigma**2 * kappa
  eq2 -= abs(sigma) * (1 - kappa)
  eq3 -= gamma

  return -np.array([eq1, eq2, eq3, eq4])


def get_system(kappa, eta, gamma, b0, use_eta=True):
  system_ = functools.partial(equations, kappa, eta, gamma, b0, use_eta)
  return system_


def correction_factors(kappa, eta, gamma, b0, use_eta=True):
  """Computes correction factors for MLE of high dimensional logistic reg."""
  system_ = get_system(kappa, eta, gamma, b0, use_eta)
  if use_eta:
    init = np.array([2, 2, np.sqrt(eta / 2), b0 / 2])
  else:
    init = np.array([2, 2, np.sqrt(gamma**2 + 1), b0])
  soln = scipy.optimize.root(
      lambda x: system_(*x),
      init,
      method='lm',
      options={
          'xtol': 1e-4,
          'eps': 1e-8
      })
  x0 = soln.x
  if kappa >= 0.03 and (x0[0] < 1 or x0[2] < 0.1):
    print('Rerunning due to convergence issue')
    init += 0.1 * np.random.randn(4)
    init = np.maximum(init, np.array([1, 0.5, 0.1, b0 / 2.0]))
    soln = scipy.optimize.root(
        lambda x: system_(*x),
        init,
        method='lm',
        options={
            'xtol': 1e-4,
            'eps': 1e-8
        })
    x0 = soln.x
  return x0


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  sol = correction_factors(0.2, 1, np.sqrt(5), 0, use_eta=False)
  print(sol)
  sol = correction_factors(0.1, 8.881028475794636, np.sqrt(5), 0, use_eta=True)
  print(sol)


if __name__ == '__main__':
  app.run(main)
