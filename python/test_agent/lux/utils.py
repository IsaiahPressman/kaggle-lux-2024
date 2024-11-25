# ruff: noqa: RET505
from typing import Sequence


# direction (0 = center, 1 = up, 2 = right, 3 = down, 4 = left)
def direction_to(src: Sequence[int], target: Sequence[int]) -> int:
    dx = target[0] - src[0]
    dy = target[1] - src[1]
    if dx == 0 and dy == 0:
        return 0

    if abs(dx) > abs(dy):
        if dx > 0:
            return 2
        else:
            return 4
    else:
        if dy > 0:
            return 3
        else:
            return 1
