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

"""Run experiment to understand runtime of SLOE relative to ProbeFrontier.

Tests the runtime of the SLOE estimator in compared to
ProbeFrontier over many seeds, storing in csv files to be analyzed in a colab.
"""
import time



from absl import app
from absl import flags
import apache_beam as beam
from apache_beam.options import pipeline_options
import numpy as np

import sloe_logistic.sloe_experiments.experiment_helpers as exp_helper


FLAGS = flags.FLAGS

flags.DEFINE_integer('num_sims', 100, 'number of simulations to run')
flags.DEFINE_string('output_path', '/tmp/counts.txt', 'The output file path')

N_RANGE = [500, 1000, 2000, 3000, 4000, 6000, 8000, 16000]


def run_sim(val):
  """Runs simulation and compare runtime of SLOE and ProbeFrontier."""
  n = val[0]
  seed = 201216 + val[1]
  # Model parameters

  sim_params = exp_helper.SimulationParams.create_from_flags()
  sim_params.seed = seed
  sim_params.training_n = n
  sim_params.p = int(n * FLAGS.features_per_sample)
  sim = exp_helper.Simulation(sim_params)

  new_method_model = exp_helper.create_inference_model('newmethod')
  pf_model = exp_helper.create_inference_model('probe_frontier')

  x1, y1 = sim.sample()
  if pf_model.is_separable(x1, y1):
    return

  tic = time.perf_counter()
  m = new_method_model.fit(x1, y1)
  toc = time.perf_counter()
  new_method_time = toc - tic
  # Deleting model here to keep memory clean for probe frontier model.
  del new_method_model, m

  tic = time.perf_counter()
  m, v = pf_model.fit(x1, y1)
  toc = time.perf_counter()
  probe_frontier_time = toc - tic

  return [np.array([n, seed, new_method_time, probe_frontier_time, v])]


def main(unused_argv):
  # If you have custom beam options add them here.
  beam_options = pipeline_options.PipelineOptions()

  with beam.Pipeline(options=beam_options) as pipe:
    _ = (
        pipe
        | beam.Create(range(FLAGS.num_sims))
        | beam.FlatMap(exp_helper.multiple_sample_sizes, N_RANGE)
        | 'PrepShuffle' >> beam.Reshuffle()
        | beam.FlatMap(run_sim)
        | beam.Map(exp_helper.numpy_array_to_csv)
        | beam.Reshuffle()
        |
        'WriteToText' >> beam.io.WriteToText(FLAGS.output_path, num_shards=5))


if __name__ == '__main__':
  app.run(main)
