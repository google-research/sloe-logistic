// Copyright 2021 The SLOE Logistic Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mle_param_integrands.h"

#include <iostream>

#include "pybind11/pybind11.h"

namespace logistic_hd {

double sigmoid(double z) {
  const double v = 1.0 / (1 + exp(-z));
  return (v);
}

double prox_deriv(double z, void *args) {
  prox_params *myargs = reinterpret_cast<prox_params *>(args);
  return (myargs->lambda * sigmoid(z) + z - myargs->x);
}

double prox_impl(double lambda, double x, double xtol, double rtol,
                 int maxiters) {
  prox_params params;
  scipy_zeros_info solver_stats;
  double lower;
  double upper;

  params.lambda = lambda;
  params.x = x;

  if (lambda * x > 0) {
    lower = x - lambda - 1e-4;
    upper = x + 1e-4;
  } else {
    lower = x - lambda / 2.0 - 1e-4;
    upper = x + 1e-4;
  }
  lower = -abs(x) - 8;
  upper = abs(x) + 8;

  if (abs(prox_deriv(lower, &params)) < 1e-8) {
    return (lower);
  }
  if (abs(prox_deriv(upper, &params)) < 1e-8) {
    return (upper);
  }

  const double x0 = brentq(&prox_deriv, lower, upper, xtol, rtol, maxiters,
                           reinterpret_cast<void *>(&params), &solver_stats);

  return (x0);
}

double integrand(double Z1, double Z2, double kappa, double gamma, double b0,
                 double alpha, double lambda, double sigma, double beta0,
                 int eq_num) {
  double eq;

  const double S1 = gamma * Z1 / alpha + beta0;
  const double S2 = gamma * Z1 + sigma * Z2 + b0;

  const double prox_S2 = prox_impl(lambda, S2);
  const double prox_lambda_S2 = prox_impl(lambda, lambda + S2);

  const double sig_S1 = sigmoid(S1);
  const double sig_neg_S1 = 1 - sig_S1;

  if (eq_num == 1) {
    eq = sig_S1 * pow(S2 - prox_lambda_S2, 2);
    eq += sig_neg_S1 * pow(S2 - prox_S2, 2);
  } else if (eq_num == 2) {
    eq = sig_S1 * Z2 * prox_lambda_S2;
    eq += sig_neg_S1 * Z2 * prox_S2;
  } else if (eq_num == 3) {
    eq = sig_S1 * Z1 * prox_lambda_S2;
    eq += sig_neg_S1 * Z1 * prox_S2;
  } else {
    const double prox_neg_S2 = prox_impl(lambda, -S2);
    eq = -sig_S1 * sigmoid(prox_neg_S2);
    eq += sig_neg_S1 * sigmoid(prox_S2);
  }

  return (eq * pdf(Z1, Z2));
}

double pdf(double x1, double x2) {
  return (exp(-(pow(x1, 2) + pow(x2, 2)) / 2.0) / (2 * M_PI));
}

}  // namespace logistic_hd

PYBIND11_MODULE(mle_param_integrands, m) {
  m.doc() = "Logistic Regression MLE High Dimensional Integrands";

  m.def("sigmoid", &logistic_hd::sigmoid,
        "Sigmoid for a float (unvectorized, no error checking)");
  m.def("integrand", &logistic_hd::integrand,
        "Integrand for equation to get high dimensional adjustment");
  m.def("prox_deriv", &logistic_hd::prox_deriv,
        "Derivative prox objective for logistic link");
  m.def("prox_impl", &logistic_hd::prox_impl,
        "Computes prox for logistic link times lambda");
  m.def("pdf", &logistic_hd::pdf,
        "Computes pdf of bivariate normal distribution");
}
