from __future__ import annotations

import sys
import time
from enum import Enum
from functools import partial

from loguru import logger

from game_input import click, press
from recognition import locate
from towers import Hero, Ninja, Sub
from utils import Box, Point, padding, screen_size, sleep


# positions mapped at 2560x1440
class COORDS(Point, Enum):
    COLLECTION_EVENT_COLLECT = 1279, 911
    COLLECTION_EVENT_CONTINUE = 1280, 1330
    COLLECTION_EVENT_INSTA_L = 1075, 750
    COLLECTION_EVENT_INSTA_R = 1480, 750
    GAME_DIFFICULTY_EASY = 838, 550
    GAME_MODE_STANDARD = 847, 780
    HERO_OBYN = 135, 550
    HERO_SELECT = 1490, 815
    HOME_HEROES = 800, 1265
    HOME_PLAY = 1123, 1248
    MAP_DARK_CASTLE = 730, 780
    MAP_DIFFICULTY_EXPERT = 1778, 1304
    OVERWRITE_SAVE_YES = 1520, 974
    TOWER_HERO = 738, 600
    TOWER_NINJA = 738, 844
    TOWER_SUB = 1454, 575
    VICTORY_CONTINUE = 1283, 1215
    VICTORY_HOME = 957, 1135


# position of each screenshot at 2560x1440
class IMAGE_BOXES(Box, Enum):
    COLLECT = 1116, 849, 322, 114
    MENU = 45, 594, 119, 101
    OBYN = 717, 1219, 188, 130
    PLAY = 2206, 1281, 145, 148
    VICTORY = 943, 187, 668, 116


locate_collect = partial(locate, "collect.png", region=padding(IMAGE_BOXES.COLLECT))
locate_menu = partial(locate, "menu.png", region=IMAGE_BOXES.MENU)
locate_obyn = partial(locate, "obyn.png", region=padding(IMAGE_BOXES.OBYN))
locate_play_button = partial(locate, "play_button.png", region=padding(IMAGE_BOXES.PLAY))
locate_victory = partial(locate, "victory.png", region=padding(IMAGE_BOXES.VICTORY))


def check_obyn() -> None:
    logger.info("Checking for Obyn")
    if locate_obyn():
        return

    logger.info("Obyn not selected, changing hero")
    click(COORDS.HOME_HEROES)
    time.sleep(0.2)
    click(COORDS.HERO_OBYN, add_padding=False)
    click(COORDS.HERO_SELECT)
    press("esc")

    assert locate_obyn(0.5)


def check_collection_event() -> None:
    if not locate_collect():
        return

    logger.info("Collection event detected")
    click(COORDS.COLLECTION_EVENT_COLLECT)
    time.sleep(1)

    click(COORDS.COLLECTION_EVENT_INSTA_L, clicks=2, interval=1)
    time.sleep(0.5)
    click(COORDS.COLLECTION_EVENT_INSTA_R, clicks=2, interval=1)
    time.sleep(0.5)

    click(COORDS.COLLECTION_EVENT_CONTINUE)
    time.sleep(0.2)
    press("esc")
    time.sleep(2)


def select_map() -> None:
    logger.info("Selecting map")

    click(COORDS.HOME_PLAY)
    time.sleep(0.1)
    click(COORDS.MAP_DIFFICULTY_EXPERT, clicks=2)
    click(COORDS.MAP_DARK_CASTLE)
    click(COORDS.GAME_DIFFICULTY_EASY)
    click(COORDS.GAME_MODE_STANDARD)
    time.sleep(0.1)
    click(COORDS.OVERWRITE_SAVE_YES)


def play_game() -> None:
    logger.info("Starting game")

    sleep(2)
    if not locate_play_button(3):
        raise Exception("Play button not detected")

    time.sleep(0.2)
    # Start and fast-forward the game
    press("space", presses=2)

    Hero(COORDS.TOWER_HERO)

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
    check_collection_event()

    if not locate_menu():
        raise Exception("BTD6 menu not detected")


@logger.catch(onerror=lambda _: sys.exit(1))
def main() -> None:
    if screen_size().height != 1440:
        logger.error("Unsupported resolution")
        return

    logger.info("Focus the BTD6 window within 5 seconds")
    if not locate_menu(5):
        logger.error("BTD6 menu not detected")
        return
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
