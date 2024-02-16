from __future__ import annotations

import sys
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
import pyscreeze
from cv2.typing import MatLike
from fast_ctypes_screenshots import ScreenshotOfOneMonitor
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


# position of each screenshot at 2560x1440
class IMAGE_BOXES(pyscreeze.Box, Enum):
    MENU = pyscreeze.Box(45, 594, 119, 101)
    OBYN = pyscreeze.Box(717, 1219, 188, 130)
    VICTORY = pyscreeze.Box(943, 187, 668, 116)


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

pyscreeze.USE_IMAGE_NOT_FOUND_EXCEPTION = False

# Determine which pictures are loaded (pictures are resolution-specific!)
screen_width, screen_height = pyautogui.size()

IMAGES = {
    name: cv2.imread(str(Path.cwd() / "images" / f"{screen_height}_{name}.png"), flags=cv2.IMREAD_GRAYSCALE)
    for name in ["victory", "defeat", "menu", "easter", "obyn"]
}

reso_16_9: list[tuple[int, int]] = [(1920, 1080), (2560, 1440), (3840, 2160)]


@contextmanager
def timer_ns() -> Generator[Callable[[], int], Any, None]:
    t1 = t2 = time.time_ns()
    yield lambda: t2 - t1
    t2 = time.time_ns()


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


def locate_on_screen(
    image: MatLike,
    min_search_time: float = 0,
    *,
    region: tuple[int, int, int, int] | None = None,
    confidence: float = 0.999,
) -> pyscreeze.Box | None:
    """Similar to `pyscreeze.locateOnScreen()`, but uses `fast-ctypes-screenshots` for screenshots."""
    with ScreenshotOfOneMonitor() as sm:
        start_time = time.monotonic()
        while True:
            with timer_ns() as t:
                screenshot = sm.screenshot_one_monitor()
            logger.debug(f"Screenshot took {t() / 1e6} ms")

            with timer_ns() as t:
                box = pyscreeze.locate(image, screenshot, grayscale=True, region=region, confidence=confidence)
            logger.debug(f"pyscreeze.locate() took {t() / 1e6} ms")

            if box is not None:
                return box
            elif time.monotonic() - start_time > min_search_time:
                return None


def locate(name: str, min_search_time: float = 0, *, region: tuple[int, int, int, int] | None = None) -> bool:
    with timer_ns() as t:
        box = locate_on_screen(IMAGES[name], min_search_time, region=region, confidence=0.9)

    if box is None:
        logger.debug(f"{name} not found after {t() / 1e6} ms")
        return False
    else:
        logger.debug(f"{name} located at {box} in {t() / 1e6} ms")
        return True


def calculate_search_region(region_2560_1440: pyscreeze.Box, add_padding: bool = True) -> pyscreeze.Box:
    left, top = scaling(region_2560_1440.left, region_2560_1440.top, add_padding=add_padding)
    return pyscreeze.Box(
        left := max(0, left - region_2560_1440.width),
        top := max(0, top - region_2560_1440.height),
        min(screen_width - left, region_2560_1440.width * 2),
        min(screen_height - top, region_2560_1440.height * 2),
    )


def locate_menu(min_search_time: float = 0) -> bool:
    search_region = calculate_search_region(IMAGE_BOXES.MENU, add_padding=False)
    return locate("menu", min_search_time, region=search_region)


def locate_obyn(min_search_time: float = 0) -> bool:
    search_region = calculate_search_region(IMAGE_BOXES.OBYN)
    return locate("obyn", min_search_time, region=search_region)


def locate_victory(min_search_time: float = 0) -> bool:
    search_region = calculate_search_region(IMAGE_BOXES.VICTORY)
    return locate("victory", min_search_time, region=search_region)


def obyn_check() -> None:
    logger.info("Checking for Obyn")
    if locate_obyn():
        return

    logger.info("Obyn not selected, changing hero")
    click(COORDS.HERO_SELECT)
    click(COORDS.HERO_OBYN, add_padding=False)
    click(COORDS.HERO_CONFIRM)
    press("esc")

    assert locate_obyn()


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

    sleep(3)


def main_game() -> None:
    logger.info("Starting game")

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

    if locate_victory(min_search_time=5):
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
    if not locate_menu(min_search_time=5):
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
