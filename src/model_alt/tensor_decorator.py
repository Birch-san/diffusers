from torch import Tensor
from typing import Callable, TypeVar
from typing_extensions import TypeAlias

T = TypeVar('T')
Decorator: TypeAlias = Callable[[T], T]
TensorDecorator: TypeAlias = Decorator[Tensor]