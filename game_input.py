from __future__ import annotations

import time
from typing import overload

import pydirectinput
from pydirectinput import MOUSE_PRIMARY

from utils import Point, padding

# Default time in seconds to pause after every public function call
PAUSE = 0.1


@overload
def click(*, clicks: int = 1, interval: float = 0.1, button: str = MOUSE_PRIMARY, pause: bool = True) -> None: ...


@overload
def click(
    coords: Point,
    *,
    add_padding: bool = True,
    clicks: int = 1,
    interval: float = 0.1,
    button: str = MOUSE_PRIMARY,
    pause: bool = True,
) -> None: ...


def click(
    coords: Point | None = None,
    *,
    add_padding: bool = True,
    clicks: int = 1,
    interval: float = 0.1,
    button: str = MOUSE_PRIMARY,
    pause: bool = True,
) -> None:
    if coords is None:
        pydirectinput.click(clicks=clicks, interval=interval)
    else:
        pydirectinput.click(
            *(padding(coords) if add_padding else coords), clicks=clicks, interval=interval, button=button
        )

    if pause:
        time.sleep(PAUSE)


def move_to(coords: Point, *, add_padding: bool = True, pause: bool = True) -> None:
    pydirectinput.moveTo(*(padding(coords) if add_padding else coords))

    if pause:
        time.sleep(PAUSE)


def press(key: str, *, presses: int = 1, interval: float = 0.1, pause: bool = True) -> None:
    pydirectinput.press(key, presses=presses, interval=interval)

    if pause:
        time.sleep(PAUSE)
