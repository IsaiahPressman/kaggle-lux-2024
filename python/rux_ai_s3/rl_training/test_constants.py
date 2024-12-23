from .constants import TRAIN_OUTPUTS_DIR


def test_train_outputs() -> None:
    assert TRAIN_OUTPUTS_DIR.is_dir()
