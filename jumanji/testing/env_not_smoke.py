# Copyright 2022 InstaDeep Ltd. All rights reserved.
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

from typing import Callable, Optional

import chex
import jax

from jumanji.env import Environment
from jumanji.types import Observation

SelectActionFn = Callable[[chex.PRNGKey, Observation], chex.Array]


def check_env_does_not_smoke(
    env: Environment,
    select_action: Optional[SelectActionFn] = None,
    assert_finite_check: bool = True,
) -> None:
    """Run an episode of the environment, with a jitted step function to check no errors occur."""
    if select_action is None:
        select_action = env.sample_action

    key = jax.random.PRNGKey(0)
    key, reset_key = jax.random.split(key)
    state, timestep = env.reset(reset_key)
    step_fn = jax.jit(env.step)
    while not timestep.last():
        key, action_key = jax.random.split(key)
        action = select_action(action_key, timestep.observation)
        env.action_spec.validate(action)
        state, timestep = step_fn(state, action)
        env.observation_spec.validate(timestep.observation)
        if assert_finite_check:
            chex.assert_tree_all_finite((state, timestep))


def access_specs(env: Environment) -> None:
    """Access specs of the environment."""
    env.observation_spec
    env.action_spec
    env.reward_spec
    env.discount_spec


def check_env_specs_does_not_smoke(env: Environment) -> None:
    """Access specs of the environment in a jitted function to check no errors occur."""
    jax.jit(access_specs, static_argnums=0)(env)
