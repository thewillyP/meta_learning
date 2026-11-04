from dataclasses import dataclass
from typing import Callable


type Unit = tuple[()]


@dataclass(frozen=True)
class Proxy[T]: ...


@dataclass(frozen=True)
class Monoid[P, DP]:
    mempty: Callable[[P], DP]
    mappend: Callable[[DP, DP], DP]
