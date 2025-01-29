from pathlib import Path
from typing import Final

PROJECT_NAME: Final[str] = "rux_ai_s3"
TRAIN_OUTPUTS_DIR: Final[Path] = Path(__file__).parents[3] / "train_outputs"
TRAIN_CONFIG_FILE_NAME: Final[str] = "train_config.yaml"
