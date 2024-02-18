from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

from loguru import logger

from game_input import click, move_to, press
from utils import Point


class Tower(ABC):
    coords: Point
    upgrades: list[int]

    def __init__(self, coords: Point) -> None:
        self.coords = coords
        self.upgrades = [0] * 3

        logger.info(f"Placing down {type(self).__name__}")
        move_to(self.coords, pause=False)
        press(self.hotkey, pause=False)
        click(pause=False)

    def __str__(self) -> str:
        return f"{''.join(map(str, self.upgrades))} {type(self).__name__} at {self.coords}"

    @property
    @abstractmethod
    def hotkey(self) -> str:
        return NotImplemented

    def upgrade(self, path: Literal[1, 2, 3]) -> None:
        name_before_upgrade = str(self)
        self.upgrades[path - 1] += 1
        logger.info(f"Upgrading {name_before_upgrade} to {''.join(map(str, self.upgrades))}")

        click(self.coords)
        press(["", ",", ".", "/"][path])
        press("esc")


class Obyn(Tower):
    @property
    def hotkey(self) -> str:
        return "u"


class Ninja(Tower):
    @property
    def hotkey(self) -> str:
        return "d"


class Sub(Tower):
    @property
    def hotkey(self) -> str:
        return "x"
