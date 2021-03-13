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

"""Builds sloe_logistic package."""

from distutils import core
from distutils.command import build_clib

from pybind11.setup_helpers import build_ext
from pybind11.setup_helpers import Pybind11Extension

libraries = [
    ("scipy_brentq", {
        "sources": ["third_party/py/scipy/optimize/Zeros/brentq.c",],
    }),
]

ext_modules = [
    Pybind11Extension("sloe_logistic.mle_param_integrands", [
        "mle_param_integrands.cc",
    ]),
]

core.setup(
    name="sloe_logistic",
    version="0.0.1",
    description="Implements SLOE method and Logistic Regression Inference",
    long_description="Code to supplement the ICML submission SLOE: A Faster "
    "Method for Statistical Inference in High-Dimensional Logistic Regression.",
    packages=["sloe_logistic", "sloe_logistic.sloe_experiments"],
    package_dir={
        "sloe_logistic": ".",
        "sloe_logistic.sloe_experiments": "sloe_experiments/"
    },
    libraries=libraries,
    ext_modules=ext_modules,
    cmdclass={
        "build_ext": build_ext,
        "build_clib": build_clib.build_clib,
    },
    zip_safe=False,
)
