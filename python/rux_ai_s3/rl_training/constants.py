from pathlib import Path
from typing import Final

MAP_SIZE: Final[int] = 24
PROJECT_NAME: Final[str] = "rux_ai_s3"
TRAIN_OUTPUTS_DIR: Final[Path] = Path(__file__).parents[3] / "train_outputs"
