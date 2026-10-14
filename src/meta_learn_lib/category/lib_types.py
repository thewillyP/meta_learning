from dataclasses import dataclass


type Unit = tuple[()]


@dataclass(frozen=True)
class Proxy[T]: ...
