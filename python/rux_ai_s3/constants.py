from enum import Enum
from typing import Final

MAP_SIZE: Final[int] = 24
PROJECT_NAME: Final[str] = "rux_ai_s3"


class Action(Enum):
    NO_OP = 0
    UP = 1
    RIGHT = 2
    DOWN = 3
    LEFT = 4
    SAP = 5
