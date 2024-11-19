import numpy as np
import numpy.typing as npt

def hello_world() -> str: ...
def hello_numpy_world() -> npt.NDArray[np.float32]: ...

class ParallelEnv:
    def __init__(self, n_envs: int) -> None: ...
    def soft_reset(
        self,
        tile_type: npt.NDArray[np.int32],
        energy_nodes: npt.NDArray[np.int16],
        energy_node_fns: npt.NDArray[np.float32],
        energy_nodes_mask: npt.NDArray[np.bool_],
        relic_nodes: npt.NDArray[np.int16],
        relic_node_configs: npt.NDArray[np.bool_],
        relic_nodes_mask: npt.NDArray[np.bool_],
    ) -> None: ...
    def seq_step(
        self, actions: npt.NDArray[np.int_]
    ) -> tuple[npt.NDArray[np.float32], list[tuple[int, int]], list[bool]]: ...
