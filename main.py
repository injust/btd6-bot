from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Generator
from contextlib import contextmanager
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import cv2
import pyautogui
import pydirectinput
from loguru import logger


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


class Tower(ABC):
    coords: COORDS
    upgrades: list[int]

    def __init__(self, coords: COORDS) -> None:
        self.coords = coords
        self.upgrades = [0] * 3

        logger.info(f"Placing down {type(self).__name__}")
        pyautogui.moveTo(scaling(*self.coords))
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

# Determine which pictures are loaded (pictures are resolution-specific!)
screen_height = pyautogui.size()[1]

IMAGES = {
    name: cv2.imread(str(Path.cwd() / "images" / f"{screen_height}_{name}.png"), flags=cv2.IMREAD_GRAYSCALE)
    for name in ["victory", "defeat", "menu", "easter", "obyn"]
}

reso_16_9: list[tuple[int, int]] = [(1920, 1080), (2560, 1440), (3840, 2160)]


@contextmanager
def timer() -> Generator[Callable[[], float], Any, None]:
    t1 = t2 = time.monotonic()
    yield lambda: t2 - t1
    t2 = time.monotonic()


def scaling(x: int, y: int, add_padding: bool = True) -> tuple[int, int]:
    """Adjust coordinates from 2560x1440 to current resolution and add padding to account for 21:9."""
    width, height = pyautogui.size()
    for res in reso_16_9:
        if height == res[1]:
            padding = (width - res[0]) // 2
            if width == res[0]:
                # 16:9 resolution
                x = x * width // 2560
                break
            else:
                # 21:9 resolution
                break
    else:
        raise Exception("Unsupported resolution")
    y = y * height // 1440
    if add_padding:
        x += padding
    return x, y


def click(coords: COORDS, add_padding: bool = True) -> None:
    pyautogui.click(scaling(*coords, add_padding=add_padding))
    time.sleep(0.2)


def press(key: str) -> None:
    pydirectinput.press(key)
    time.sleep(0.2)


def sleep(seconds: float) -> None:
    logger.debug(f"Sleeping for {seconds} seconds")
    time.sleep(seconds)


def locate(name: str, search_time: float = 0) -> bool:
    try:
        box = pyautogui.locateOnScreen(IMAGES[name], minSearchTime=search_time, grayscale=True, confidence=0.9)
        logger.debug(f"{name} located at {box}")
        return True
    except pyautogui.ImageNotFoundException:
        return False


def obyn_check() -> None:
    if locate("obyn"):
        return

    logger.info("Obyn not selected, changing hero")
    click(COORDS.HERO_SELECT)
    click(COORDS.HERO_OBYN, add_padding=False)
    click(COORDS.HERO_CONFIRM)
    press("esc")

    assert locate("obyn")


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
    assert locate("menu")

    click(COORDS.HOME_START)
    click(COORDS.EXPERT_SELECTION)
    click(COORDS.EXPERT_SELECTION)
    click(COORDS.DARK_CASTLE)
    click(COORDS.EASY_DIFFICULTY)
    click(COORDS.STANDARD_MODE)
    click(COORDS.OVERWRITE_SAVE)


def main_game() -> None:
    logger.info("Starting game")
    sleep(3)

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
    sleep(55.5)


def exit_game() -> None:
    logger.info("Game ending, returning to menu")

    if locate("victory", search_time=5):
        click(COORDS.VICTORY_CONTINUE)
        time.sleep(0.2)
    elif not locate("defeat"):
        raise Exception("Victory/Defeat not detected")

    click(COORDS.VICTORY_HOME)
    time.sleep(2)
    easter_event_check()
    assert locate("menu")


###########################################[MAIN LOOP]###########################################
logger.info("Focus BTD6 window within 5 seconds")
if not locate("menu", search_time=5):
    raise Exception("BTD6 window not detected")
time.sleep(0.5)

obyn_check()

while True:
    select_map()
    main_game()
    exit_game()


###########################################U
