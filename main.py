from __future__ import annotations

import sys
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Generator
from contextlib import contextmanager
from enum import Enum
from pathlib import Path
from typing import Any, Literal, overload

import cv2
import numpy as np
import pyautogui
import pydirectinput
import pyscreeze
from cv2.typing import MatLike
from loguru import logger
from mss import mss


# positions mapped at 2560x1440
class COORDS(tuple[int, int], Enum):
    HOME_START = (1123, 1248)
    EXPERT_SELECTION = (1778, 1304)
    DARK_CASTLE = (730, 780)
    EASY_DIFFICULTY = (838, 550)
    STANDARD_MODE = (847, 780)
    OVERWRITE_SAVE = (1520, 974)
    TOWER_OBYN = (738, 600)
    TOWER_SUB = (1454, 575)
    TOWER_NINJA = (738, 844)
    VICTORY_CONTINUE = (1283, 1215)
    VICTORY_HOME = (957, 1135)
    EASTER_COLLECTION = (1279, 911)
    EASTER_INSTA_L = (1075, 750)
    EASTER_INSTA_R = (1480, 750)
    EASTER_CONTINUE = (1280, 1330)
    HERO_SELECT = (800, 1265)
    HERO_OBYN = (135, 550)
    HERO_CONFIRM = (1490, 815)


# position of each screenshot at 2560x1440
class IMAGE_BOXES(pyscreeze.Box, Enum):
    MENU = pyscreeze.Box(45, 594, 119, 101)
    OBYN = pyscreeze.Box(717, 1219, 188, 130)
    PLAY = pyscreeze.Box(2206, 1281, 145, 148)
    VICTORY = pyscreeze.Box(943, 187, 668, 116)


class Tower(ABC):
    coords: COORDS
    upgrades: list[int]

    def __init__(self, coords: COORDS) -> None:
        self.coords = coords
        self.upgrades = [0] * 3

        logger.info(f"Placing down {type(self).__name__}")
        pyautogui.moveTo(padding(self.coords))
        press(self.hotkey)
        pyautogui.click()
        time.sleep(0.1)

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


###########################################[SETUP]###########################################

if pyautogui.size().height != 1440:
    raise Exception("Unsupported resolution")

IMAGES = {
    path.stem: cv2.imread(str(path), flags=cv2.IMREAD_GRAYSCALE) for path in (Path.cwd() / "images").glob("*.png")
}


@contextmanager
def timer_ns() -> Generator[Callable[[], int], Any, None]:
    t1 = t2 = time.time_ns()
    yield lambda: t2 - t1
    t2 = time.time_ns()


@overload
def padding(coords: tuple[int, int]) -> tuple[int, int]: ...


@overload
def padding(coords: pyscreeze.Box) -> pyscreeze.Box: ...


def padding(coords: tuple[int, int] | pyscreeze.Box) -> tuple[int, int] | pyscreeze.Box:
    """Add padding to support 3440×1440."""
    padding = (pyautogui.size().width - 2560) // 2

    if isinstance(coords, pyscreeze.Box):
        return coords._replace(left=coords.left + padding)
    return coords[0] + padding, coords[1]


def click(coords: COORDS, add_padding: bool = True) -> None:
    pyautogui.click(padding(coords) if add_padding else coords)
    time.sleep(0.2)


def press(key: str) -> None:
    pydirectinput.press(key)
    time.sleep(0.2)


def sleep(seconds: float) -> None:
    logger.debug(f"Sleeping for {seconds} seconds")
    time.sleep(seconds)


def locate_opencv(needle: MatLike, haystack: MatLike) -> tuple[pyscreeze.Box, float]:
    """Simplified version of `pyscreeze._locateAll_opencv()."""
    # Avoid semi-cryptic OpenCV error if bad size
    if haystack.shape[0] < needle.shape[0] or haystack.shape[1] < needle.shape[1]:
        raise ValueError("Needle image dimensions exceed haystack image dimensions")

    results = cv2.matchTemplate(haystack, needle, cv2.TM_CCOEFF_NORMED)
    _, confidence, _, coords = cv2.minMaxLoc(results)
    return pyscreeze.Box(*coords, *needle.shape), confidence


def locate_on_screen(
    image: MatLike, min_search_time: float = 0, *, region: pyscreeze.Box | None = None, min_confidence: float = 0.999
) -> pyscreeze.Box | None:
    """Similar to `pyscreeze.locateOnScreen()`, but uses `mss` for screenshots and only screenshots the search region."""
    if region is None:
        region = pyscreeze.Box(0, 0, *pyautogui.size())

    with mss() as sct:
        start_time = time.monotonic()
        while True:
            with timer_ns() as t:
                screenshot = cv2.cvtColor(np.asarray(sct.grab(region._asdict())), cv2.COLOR_BGRA2GRAY)
            logger.debug(f"Screenshot took {t() / 1e6} ms")

            with timer_ns() as t:
                box, confidence = locate_opencv(image, screenshot)
            logger.debug(f"locate_opencv() took {t() / 1e6} ms, confidence={confidence:.3f}")

            if confidence >= min_confidence:
                # Adjust box to return coordinates relative to screen instead of search region
                return box._replace(left=box.left + region.left, top=box.top + region.top)
            elif time.monotonic() - start_time > min_search_time:
                return None


def locate(name: str, min_search_time: float = 0, *, region: pyscreeze.Box | None = None) -> bool:
    with timer_ns() as t:
        box = locate_on_screen(IMAGES[name], min_search_time, region=region, min_confidence=0.9)

    if box is None:
        logger.debug(f"{name} not found after {t() / 1e6} ms")
        return False
    else:
        logger.debug(f"{name} located at {box} in {t() / 1e6} ms")
        return True


def locate_menu(min_search_time: float = 0) -> bool:
    return locate("menu", min_search_time, region=IMAGE_BOXES.MENU)


def locate_obyn(min_search_time: float = 0) -> bool:
    return locate("obyn", min_search_time, region=padding(IMAGE_BOXES.OBYN))


def locate_play(min_search_time: float = 0) -> bool:
    return locate("play", min_search_time, region=padding(IMAGE_BOXES.PLAY))


def locate_victory(min_search_time: float = 0) -> bool:
    return locate("victory", min_search_time, region=padding(IMAGE_BOXES.VICTORY))


def obyn_check() -> None:
    logger.info("Checking for Obyn")
    if locate_obyn():
        return

    logger.info("Obyn not selected, changing hero")
    click(COORDS.HERO_SELECT)
    click(COORDS.HERO_OBYN, add_padding=False)
    click(COORDS.HERO_CONFIRM)
    press("esc")

    assert locate_obyn(0.5)


def easter_event_check() -> None:
    if not locate("easter"):
        return

    logger.info("Easter event detected")
    click(COORDS.EASTER_COLLECTION)
    time.sleep(1)

    click(COORDS.EASTER_INSTA_L)
    time.sleep(1)
    click(COORDS.EASTER_INSTA_L)
    time.sleep(1)
    click(COORDS.EASTER_INSTA_R)
    time.sleep(1)
    click(COORDS.EASTER_INSTA_R)
    time.sleep(2)

    click(COORDS.EASTER_CONTINUE)
    time.sleep(1)
    press("esc")
    time.sleep(2)


###########################################


###########################################[GAME]###########################################
def select_map() -> None:
    logger.info("Selecting map")

    click(COORDS.HOME_START)
    click(COORDS.EXPERT_SELECTION)
    click(COORDS.EXPERT_SELECTION)
    click(COORDS.DARK_CASTLE)
    click(COORDS.EASY_DIFFICULTY)
    click(COORDS.STANDARD_MODE)
    click(COORDS.OVERWRITE_SAVE)


def main_game() -> None:
    logger.info("Starting game")

    sleep(1.5)
    if not locate_play(3):
        raise Exception("Play button not detected")

    time.sleep(0.2)
    # Start and fast-forward the game
    pydirectinput.press("space", presses=2)

    Obyn(COORDS.TOWER_OBYN)

    sub = Sub(COORDS.TOWER_SUB)
    sub.upgrade(1)
    sleep(22)
    sub.upgrade(1)
    sleep(20.5)

    ninja = Ninja(COORDS.TOWER_NINJA)
    sleep(20)

    sub.upgrade(3)
    sleep(37.5)
    sub.upgrade(3)
    sleep(13)

    ninja.upgrade(1)
    sleep(9)
    ninja.upgrade(1)
    sleep(9)
    ninja.upgrade(3)
    sleep(13)
    ninja.upgrade(1)
    sleep(19)

    sub.upgrade(3)
    sleep(41)
    sub.upgrade(3)
    sleep(43.5)

    ninja.upgrade(1)
    sleep(55)


def exit_game() -> None:
    logger.info("Game ending, returning to menu")

    if locate_victory(5):
        click(COORDS.VICTORY_CONTINUE)
        time.sleep(0.2)
    elif not locate("defeat"):
        raise Exception("Victory/Defeat not detected")

    click(COORDS.VICTORY_HOME)
    time.sleep(2)
    easter_event_check()

    if not locate_menu():
        raise Exception("BTD6 menu not detected")


@logger.catch(onerror=lambda _: sys.exit(1))
def main() -> None:
    logger.info("Focus the BTD6 window within 5 seconds")
    if not locate_menu(5):
        raise Exception("BTD6 menu not detected")
    time.sleep(0.5)

    obyn_check()

    try:
        while True:
            select_map()
            main_game()
            exit_game()
    except KeyboardInterrupt:
        logger.debug("Shutting down")


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, diagnose=True)

    main()
