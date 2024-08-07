from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal, override

from attrs import define, field
from loguru import logger

from game_input import click, move_to, press
from utils import Point


@define
class Tower(ABC):
    coords: Point
    upgrades: list[int] = field(init=False, factory=lambda: [0] * 3)

    def __attrs_postinit__(self) -> None:
        logger.info("Placing down {}", type(self).__name__)
        move_to(self.coords, pause=False)
        press(self.hotkey, pause=False)
        click(pause=False)

    @override
    def __str__(self) -> str:
        return f"{"".join(map(str, self.upgrades))} {type(self).__name__} at {self.coords}"

    @property
    @abstractmethod
    def hotkey(self) -> str:
        return NotImplemented

    def upgrade(self, path: Literal[1, 2, 3]) -> None:
        name_before_upgrade = str(self)
        self.upgrades[path - 1] += 1
        logger.info("Upgrading {} to {}", name_before_upgrade, "".join(map(str, self.upgrades)))

        click(self.coords)
        press(["", ",", ".", "/"][path])
        press("esc")


@define
class Hero(Tower):
    @override
    def __str__(self) -> str:
        return f"{type(self).__name__} at {self.coords}"

    @property
    @override
    def hotkey(self) -> str:
        return "u"

    @override
    def upgrade(self, path: Literal[1, 2, 3]) -> None:
        raise TypeError("Heroes cannot be upgraded")


@define
class Ninja(Tower):
    @property
    @override
    def hotkey(self) -> str:
        return "d"


@define
class Sub(Tower):
    @property
    @override
    def hotkey(self) -> str:
        return "x"
