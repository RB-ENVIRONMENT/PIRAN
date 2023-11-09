from typing import Sequence


class MagField:
    def __init__(self, mlat: Sequence[float], l_shell: float) -> None:
        self.mlat = mlat
        self.l_shell = l_shell
