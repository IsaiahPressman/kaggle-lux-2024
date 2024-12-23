import itertools
from collections import deque
from typing import Any

from rux_ai_s3.lowlevel import FeatureEngineeringEnv as LowLevelEnv
from rux_ai_s3.types import ActionArray, FeatureEngineeringOut, Obs


class FeatureEngineeringEnv:
    def __init__(
        self,
        frame_stack_len: int,
        team_id: int,
        env_params: dict[str, Any],
    ) -> None:
        self.frame_stack_len = frame_stack_len
        self._env = LowLevelEnv(
            team_id=team_id,
            env_params=env_params,
        )
        self._last_out: FeatureEngineeringOut = FeatureEngineeringOut.from_raw(
            self._env.get_empty_outputs()
        )
        self._frame_history: deque[Obs] = self._make_empty_frame_history()

    @property
    def last_out(self) -> FeatureEngineeringOut:
        return self._last_out

    def get_frame_stacked_obs(self) -> Obs:
        return Obs.concatenate_frame_history(self._frame_history, axis=1)

    def _make_empty_out(self) -> FeatureEngineeringOut:
        return FeatureEngineeringOut.from_raw(self._env.get_empty_outputs())

    def _make_empty_frame_history(self) -> deque[Obs]:
        return deque(
            [self._make_empty_out().obs for _ in range(self.frame_stack_len)],
            maxlen=self.frame_stack_len,
        )

    def _update_frame_history(
        self, last_out: FeatureEngineeringOut, is_new_match: bool
    ) -> None:
        self._frame_history.append(last_out.obs)
        if not is_new_match:
            return

        # Zero out all but the latest observation
        for obs in itertools.islice(self._frame_history, len(self._frame_history) - 1):
            for array in obs:
                array[:] = 0

    def step(
        self,
        lux_obs: str,
        last_actions: ActionArray,
        is_new_match: bool,
    ) -> None:
        last_out = FeatureEngineeringOut.from_raw(self._env.step(lux_obs, last_actions))
        self._update_frame_history(last_out, is_new_match)
