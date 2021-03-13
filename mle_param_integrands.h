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

#ifndef MLE_PARAM_INTEGRANDS_H_
#define MLE_PARAM_INTEGRANDS_H_

#include <math.h>

extern "C" {
#include "third_party/py/scipy/optimize/Zeros/zeros.h"
}

namespace logistic_hd {

// Integrands for the equations defined in Eq. 5 from Sur and Candès
// (PNAS, 2019). These are called by the bivariate integration over Z1 and Z2
// in `asymp_system_solve.py`.
double integrand(double Z1, double Z2, double kappa, double gamma, double b0,
                 double alpha, double lambda, double sigma, double beta0,
                 int eq_num);

// Computes the derivative of the objective that defines the proximal operator.
// The prox operator is the value of z that makes this zero.
double prox_deriv(double z, void *args);

double sigmoid(double z);

// Computes the derivative of the prox operator for the logistic regression
// log likelihood.
double prox_impl(double lambda, double x, double xtol = 1e-8,
                 double rtol = 1e-8, int maxiters = 1000);

// Computes the pdf of the bivariate normal without any input validation
// because this is called many times during optimization.
double pdf(double x1, double x2);

// Helper function to pass values between our code and the scipy.optimize API.
double scipy_zeros_functions_func(double x, void *params);

typedef struct prox_params {
  double lambda;
  double x;
} prox_params;

}  // namespace logistic_hd

#endif  // MLE_PARAM_INTEGRANDS_H_
