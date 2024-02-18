from __future__ import annotations

import time
from functools import cache
from pathlib import Path

import cv2
import numpy as np
from cv2.typing import MatLike
from loguru import logger
from mss import mss

from utils import Box, screen_size, timer_ns


@cache
def load_image(file_name: str, *, grayscale: bool) -> MatLike:
    path = Path.cwd() / "images" / file_name
    if not path.is_file():
        raise ValueError(f"{path} does not exist")

    return cv2.imread(str(path), flags=cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR)


def match_template(needle: MatLike, haystack: MatLike) -> tuple[Box, float]:
    """Simplified version of `pyscreeze._locateAll_opencv()."""
    # Avoid semi-cryptic OpenCV error if bad size
    if haystack.shape[0] < needle.shape[0] or haystack.shape[1] < needle.shape[1]:
        raise ValueError("Needle image dimensions exceed haystack image dimensions")

    results = cv2.matchTemplate(haystack, needle, cv2.TM_CCOEFF_NORMED)
    _, confidence, _, coords = cv2.minMaxLoc(results)
    return Box(*coords, *needle.shape), confidence


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
                logger.debug(f"Screenshot took {t() / 1e6} ms")

                with timer_ns() as t:
                    box, confidence = match_template(image, screenshot)
                logger.debug(f"Template matching found {box} in {t() / 1e6} ms with confidence={confidence:.3f}")

                if confidence >= min_confidence:
                    # Adjust box to return coordinates relative to screen instead of search region
                    return box._replace(left=box.left + region.left, top=box.top + region.top)
                elif time.monotonic() - start_time > min_search_time:
                    return None

    with timer_ns() as t:
        box = locate_on_screen()

    if box is None:
        logger.debug(f"{image_name} not found after {t() / 1e6} ms")
        return False
    else:
        logger.debug(f"{image_name} located at {box} in {t() / 1e6} ms")
        return True
