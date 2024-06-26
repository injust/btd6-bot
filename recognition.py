from __future__ import annotations

import time
from functools import cache
from pathlib import Path

import cv2
import numpy as np
from cv2.typing import MatLike
from loguru import logger
from mss import mss

from utils import Box, Point, Size, screen_size, timer_ns


@cache
def load_image(file_name: str, *, grayscale: bool) -> MatLike:
    path = Path.cwd() / "images" / file_name
    if path.is_dir():
        raise IsADirectoryError(f"Is a directory: {path}")
    elif not path.is_file():
        raise FileNotFoundError(f"No such file: {path}")

    image = cv2.imread(str(path), flags=cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not load image: {path}")
    return image


def match_template(needle: MatLike, haystack: MatLike) -> tuple[Box, float]:
    """Simplified version of `pyscreeze._locateAll_opencv()."""
    # Avoid semi-cryptic OpenCV error if bad size
    if haystack.shape[0] < needle.shape[0] or haystack.shape[1] < needle.shape[1]:
        raise ValueError(f"{Size.of(needle)} needle image exceeds {Size.of(haystack)} haystack image")

    results = cv2.matchTemplate(haystack, needle, cv2.TM_CCOEFF_NORMED)
    _, confidence, _, point = cv2.minMaxLoc(results)
    coords = Point(*point)

    return Box(*coords, *Size.of(needle)), confidence


def locate(
    image_name: str,
    min_search_time: float = 0,
    *,
    grayscale: bool = True,
    region: Box | None = None,
    min_confidence: float = 0.9,
) -> bool:
    if region is None:
        region = Box(0, 0, *screen_size())

    image = load_image(image_name, grayscale=grayscale)
    color_conversion = cv2.COLOR_BGRA2GRAY if grayscale else cv2.COLOR_BGRA2BGR

    def locate_on_screen() -> Box | None:
        """Similar to `pyscreeze.locateOnScreen()`, but uses `mss` for screenshots and only screenshots the search region."""
        with mss() as sct:
            start_time = time.monotonic()
            while True:
                with timer_ns() as t:
                    screenshot = cv2.cvtColor(np.asarray(sct.grab(region._asdict())), color_conversion)
                logger.debug("Screenshot took {} ms", t() / 1e6)

                with timer_ns() as t:
                    box, confidence = match_template(image, screenshot)
                logger.debug("Template matching found {} in {} ms with confidence={:.3f}", box, t() / 1e6, confidence)

                if confidence >= min_confidence:
                    # Adjust box to return coordinates relative to screen instead of search region
                    return box._replace(left=box.left + region.left, top=box.top + region.top)
                elif time.monotonic() - start_time > min_search_time:
                    return None

    with timer_ns() as t:
        box = locate_on_screen()

    if box is None:
        logger.debug("{} not found after {} ms", image_name, t() / 1e6)
        return False
    else:
        logger.debug("{} located at {} in {} ms", image_name, box, t() / 1e6)
        return True
