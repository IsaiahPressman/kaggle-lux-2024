from typing import Callable

from torch import nn

ActivationFactory = Callable[[], nn.Module]
