from collections.abc import Generator

import jax
import pytest


@pytest.fixture(scope="session", autouse=True)
def _jax_use_cpu() -> Generator[None, None, None]:
    with jax.default_device(jax.devices("cpu")[0]):
        yield
