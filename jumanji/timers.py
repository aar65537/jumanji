import gc
import time
from typing import Any, Callable, Dict, Generic, List, Optional, Tuple, TypeVar

import chex
import jax

from jumanji.env import Environment, State

TimerState = TypeVar("TimerState")
SetupFn = Callable[..., TimerState]
BodyFn = Callable[[int, TimerState], TimerState]


class JaxTimer(Generic[TimerState]):
    def __init__(
        self,
        setup: SetupFn[TimerState],
        body: BodyFn[TimerState],
    ) -> None:
        def _loop(number: int, init_state: TimerState) -> TimerState:
            final_state = jax.lax.fori_loop(0, number, body, init_state)
            if not isinstance(final_state, type(init_state)):
                raise TypeError(
                    "Final timer state must be instance of initial timer state type."
                )
            return final_state

        self._setup = setup
        self._loop = jax.jit(_loop)

    @property
    def setup(self) -> SetupFn[TimerState]:
        return self._setup

    @property
    def loop(self) -> jax.stages.Wrapped:
        return self._loop

    def timeit(self, number: int = 1024, **kwargs: Dict[str, Any]) -> float:
        # Disable garbage collector
        gcold = gc.isenabled()
        gc.disable()
        try:
            # Bring functions into local namespace for faster lookup
            local_timer = time.perf_counter
            local_block = jax.block_until_ready
            # Make sure data is ready on device
            number = local_block(jax.device_put(number))
            init_state = local_block(jax.device_put(self.setup(**kwargs)))
            # Compile loop
            compiled_loop = self.loop.lower(number, init_state).compile()
            # Time loop
            t0 = local_timer()
            local_block(compiled_loop(number, init_state))
            timing = local_timer() - t0
        finally:
            # Enable garbage collector
            if gcold:
                gc.enable()
        return timing

    def repeat(
        self, repeat: int = 5, number: int = 1024, **kwargs: Dict[str, Any]
    ) -> List[float]:
        return [self.timeit(number, **kwargs) for _ in range(repeat)]

    def autorange(
        self, min_time: float = 0.2, **kwargs: Dict[str, Any]
    ) -> Tuple[int, float]:
        for i in range(31):
            number = 2**i
            time_taken = self.timeit(number, **kwargs)
            if time_taken >= min_time:
                break
        return (number, time_taken)


Action = TypeVar("Action")
Observation = TypeVar("Observation")
SelectActionFn = Callable[[chex.PRNGKey, Observation], Action]
EnvTimerState = Tuple[chex.PRNGKey, State, Observation]


class EnvTimer(
    JaxTimer[EnvTimerState[State, Observation]], Generic[State, Observation, Action]
):
    def __init__(
        self,
        env: Environment[State],
        select_action: Optional[SelectActionFn[Observation, Action]] = None,
    ) -> None:
        _action_spec = env.action_spec()

        def _setup(key: chex.PRNGKey) -> EnvTimerState[State, Observation]:
            if len(key.shape) == 1:
                init_key, reset_key = jax.random.split(key)
            else:
                keys = jax.vmap(jax.random.split)(key)
                init_key = keys[0, 0, :]
                reset_key = keys[:, 1, :]
            init_state, init_timestep = env.reset(reset_key)
            return (init_key, init_state, init_timestep.observation)

        def _select_action(key: chex.PRNGKey, observation: Observation) -> Action:
            if select_action is not None:
                action = select_action(key, observation)
            elif hasattr(observation, "action_mask"):
                action = _action_spec.sample(key, observation.action_mask)
            else:
                action = _action_spec.sample(key)
            return action

        def _body(
            _: int, timer_state: EnvTimerState
        ) -> EnvTimerState[State, Observation]:
            key, state, observation = timer_state
            next_key, action_key = jax.random.split(key)
            if len(state.key.shape) == 1:
                action = _select_action(action_key, observation)
            else:
                action_keys = jax.random.split(action_key, state.key.shape[0])
                action = jax.vmap(_select_action)(action_keys, observation)
            next_state, next_timestep = env.step(state, action)
            return (next_key, next_state, next_timestep.observation)

        super().__init__(_setup, _body)
