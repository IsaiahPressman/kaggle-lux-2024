import numpy as np
import numpy.typing as npt

_EnvArrayOut = tuple[
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.bool_],
    npt.NDArray[np.bool_],
]

_EnvFullOut = tuple[
    _EnvArrayOut,
    list[tuple[int, int]],
    list[bool],
]

def hello_world() -> str: ...
def hello_numpy_world() -> npt.NDArray[np.float32]: ...
def get_spatial_feature_count() -> int: ...
def get_global_feature_count() -> int: ...

class ParallelEnv:
    def __init__(self, n_envs: int) -> None: ...
    def terminate_envs(self, env_ids: list[int]) -> None: ...
    def soft_reset(
        self,
        obs_arrays: _EnvArrayOut,
        tile_type: npt.NDArray[np.int32],
        energy_nodes: npt.NDArray[np.int16],
        energy_node_fns: npt.NDArray[np.float32],
        energy_nodes_mask: npt.NDArray[np.bool_],
        relic_nodes: npt.NDArray[np.int16],
        relic_node_configs: npt.NDArray[np.bool_],
        relic_nodes_mask: npt.NDArray[np.bool_],
    ) -> None: ...
    def seq_step(self, actions: npt.NDArray[np.int_]) -> _EnvFullOut: ...
