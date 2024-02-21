from __future__ import annotations

import time
from collections.abc import Callable, Generator
from contextlib import contextmanager
from typing import Any, NamedTuple, Self, overload

import pydirectinput
from cv2.typing import MatLike
from loguru import logger


class Box(NamedTuple):
    left: int
    top: int
    width: int
    height: int

    @property
    def size(self) -> Size:
        return Size(self.width, self.height)

    @property
    def top_left(self) -> Point:
        return Point(self.left, self.top)


class Point(NamedTuple):
    x: int
    y: int

    def offset(self, x: int, y: int) -> Self:
        return self._replace(x=self.x + x, y=self.y + y)


class Size(NamedTuple):
    width: int
    height: int

    @classmethod
    def of(cls, image: MatLike) -> Self:
        return cls(*image.shape[1::-1])


@overload
def padding(coords: Point) -> Point: ...


@overload
def padding(coords: Box) -> Box: ...


def padding(coords: Point | Box) -> Point | Box:
    """Add padding to support 3440×1440."""
    padding = (screen_size().width - 2560) // 2

    if isinstance(coords, Point):
        return coords._replace(x=coords.x + padding)
    return coords._replace(left=coords.left + padding)


def screen_size() -> Size:
    return Size(*pydirectinput.size())


def sleep(seconds: float) -> None:
    logger.debug(f"Sleeping for {seconds} seconds")
    time.sleep(seconds)


@contextmanager
def timer_ns() -> Generator[Callable[[], int], Any, None]:
    t1 = t2 = time.time_ns()
    yield lambda: t2 - t1
    t2 = time.time_ns()
