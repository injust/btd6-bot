from __future__ import annotations

import sys
import time
from enum import Enum
from functools import partial

from loguru import logger

from game_input import click, press
from recognition import locate
from towers import Ninja, Obyn, Sub
from utils import Box, Point, padding, screen_size, sleep


# positions mapped at 2560x1440
class COORDS(Point, Enum):
    HOME_START = Point(1123, 1248)
    EXPERT_SELECTION = Point(1778, 1304)
    DARK_CASTLE = Point(730, 780)
    EASY_DIFFICULTY = Point(838, 550)
    STANDARD_MODE = Point(847, 780)
    OVERWRITE_SAVE = Point(1520, 974)
    TOWER_OBYN = Point(738, 600)
    TOWER_SUB = Point(1454, 575)
    TOWER_NINJA = Point(738, 844)
    VICTORY_CONTINUE = Point(1283, 1215)
    VICTORY_HOME = Point(957, 1135)
    EASTER_COLLECTION = Point(1279, 911)
    EASTER_INSTA_L = Point(1075, 750)
    EASTER_INSTA_R = Point(1480, 750)
    EASTER_CONTINUE = Point(1280, 1330)
    HERO_SELECT = Point(800, 1265)
    HERO_OBYN = Point(135, 550)
    HERO_CONFIRM = Point(1490, 815)


# position of each screenshot at 2560x1440
class IMAGE_BOXES(Box, Enum):
    MENU = Box(45, 594, 119, 101)
    OBYN = Box(717, 1219, 188, 130)
    PLAY = Box(2206, 1281, 145, 148)
    VICTORY = Box(943, 187, 668, 116)


###########################################[SETUP]###########################################

if screen_size().height != 1440:
    raise Exception("Unsupported resolution")

locate_menu = partial(locate, "menu.png", region=IMAGE_BOXES.MENU)
locate_obyn = partial(locate, "obyn.png", region=padding(IMAGE_BOXES.OBYN))
locate_play_button = partial(locate, "play_button.png", region=padding(IMAGE_BOXES.PLAY))
locate_victory = partial(locate, "victory.png", region=padding(IMAGE_BOXES.VICTORY))


def check_obyn() -> None:
    logger.info("Checking for Obyn")
    if locate_obyn():
        return

    logger.info("Obyn not selected, changing hero")
    click(COORDS.HERO_SELECT)
    time.sleep(0.2)
    click(COORDS.HERO_OBYN, add_padding=False)
    click(COORDS.HERO_CONFIRM)
    press("esc")

    assert locate_obyn(0.5)


def check_easter_event() -> None:
    if not locate("easter.png"):
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
    time.sleep(0.5)
    press("esc")
    time.sleep(2)


###########################################


###########################################[GAME]###########################################
def select_map() -> None:
    logger.info("Selecting map")

    click(COORDS.HOME_START)
    time.sleep(0.1)
    click(COORDS.EXPERT_SELECTION)
    click(COORDS.EXPERT_SELECTION)
    click(COORDS.DARK_CASTLE)
    click(COORDS.EASY_DIFFICULTY)
    click(COORDS.STANDARD_MODE)
    time.sleep(0.1)
    click(COORDS.OVERWRITE_SAVE)


def play_game() -> None:
    logger.info("Starting game")

    sleep(2)
    if not locate_play_button(3):
        raise Exception("Play button not detected")

    time.sleep(0.2)
    # Start and fast-forward the game
    press("space", presses=2)

    Obyn(COORDS.TOWER_OBYN)

    sub = Sub(COORDS.TOWER_SUB)
    time.sleep(0.1)
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
    sleep(67)


def exit_game() -> None:
    logger.info("Game ending, returning to menu")

    if locate_victory(5):
        click(COORDS.VICTORY_CONTINUE)
        time.sleep(0.5)
    elif not locate("defeat.png"):
        raise Exception("Victory/Defeat not detected")

    click(COORDS.VICTORY_HOME)
    time.sleep(2)
    check_easter_event()

    if not locate_menu():
        raise Exception("BTD6 menu not detected")


@logger.catch(onerror=lambda _: sys.exit(1))
def main() -> None:
    logger.info("Focus the BTD6 window within 5 seconds")
    if not locate_menu(5):
        raise Exception("BTD6 menu not detected")
    time.sleep(0.5)

    check_obyn()

    try:
        while True:
            select_map()
            play_game()
            exit_game()
    except KeyboardInterrupt:
        logger.debug("Shutting down")


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, diagnose=True)

    main()
