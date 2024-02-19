from __future__ import annotations

import time
from collections.abc import Callable, Generator
from contextlib import contextmanager
from typing import Any, NamedTuple, Self, overload

import cv2
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


def imwrite(file_name: str, image: MatLike, *, png_compression: int = 0) -> bool:
    return cv2.imwrite(file_name, image, [cv2.IMWRITE_PNG_COMPRESSION, png_compression])


@overload
def padding(coords: Point) -> Point: ...


@overload
def padding(coords: Box) -> Box: ...


def padding(coords: Point | Box) -> Point | Box:
    """Add padding to support 3440×1440."""
    padding = (screen_size().width - 2560) // 2

    match coords:
        case Point():
            return coords._replace(x=coords.x + padding)
        case Box():
            return coords._replace(left=coords.left + padding)


def screen_size() -> Size:
    return Size(*pydirectinput.size())


def sleep(seconds: float) -> None:
    logger.debug("Sleeping for {} seconds", seconds)
    time.sleep(seconds)


@contextmanager
def timer_ns() -> Generator[Callable[[], int], Any]:
    t1 = t2 = time.time_ns()
    yield lambda: t2 - t1
    t2 = time.time_ns()
