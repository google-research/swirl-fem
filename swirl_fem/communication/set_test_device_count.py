# Copyright 2024 The swirl_fem Authors.
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

"""Utility for setting number of JAX devices in tests."""

import os

import jax


def set_host_platform_device_count(num_devices: int) -> None:
  """Sets XLA flags for `num_devices`."""
  # Forked from jax.test_util
  prev_xla_flags = os.getenv('XLA_FLAGS')
  flags_str = prev_xla_flags or ''
  # Don't override user-specified device count, or other XLA flags.
  if 'xla_force_host_platform_device_count' not in flags_str:
    os.environ['XLA_FLAGS'] = (
        flags_str + f' --xla_force_host_platform_device_count={num_devices}'
    )
  # Clear any cached backends so new CPU backend will pick up the env var.
  jax.lib.xla_bridge.get_backend.cache_clear()
