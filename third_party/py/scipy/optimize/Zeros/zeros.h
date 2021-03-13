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

/* Written by Charles Harris charles.harris@sdl.usu.edu */

/* Modified to not depend on Python everywhere by Travis Oliphant.
 */

#ifndef ZEROS_H
#define ZEROS_H

typedef struct {
    int funcalls;
    int iterations;
    int error_num;
} scipy_zeros_info;


/* Must agree with _ECONVERGED, _ESIGNERR, _ECONVERR  in zeros.py */
#define CONVERGED 0
#define SIGNERR -1
#define CONVERR -2
#define EVALUEERR -3
#define INPROGRESS 1

typedef double (*callback_type)(double, void*);
typedef double (*solver_type)(callback_type, double, double, double, double,
                              int, void *, scipy_zeros_info*);

extern double bisect(callback_type f, double xa, double xb, double xtol,
                     double rtol, int iter, void *func_data,
                     scipy_zeros_info *solver_stats);
extern double ridder(callback_type f, double xa, double xb, double xtol,
                     double rtol, int iter, void *func_data,
                     scipy_zeros_info *solver_stats);
extern double brenth(callback_type f, double xa, double xb, double xtol,
                     double rtol, int iter, void *func_data,
                     scipy_zeros_info *solver_stats);
extern double brentq(callback_type f, double xa, double xb, double xtol,
                     double rtol, int iter, void *func_data,
                     scipy_zeros_info *solver_stats);

#endif
