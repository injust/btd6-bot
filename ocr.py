from __future__ import annotations

from collections.abc import Sequence
from contextlib import AbstractContextManager, ExitStack
from pathlib import Path
from types import TracebackType
from typing import Self, cast, override

import cv2
import numpy as np
from attrs import define, field
from cv2.typing import MatLike
from loguru import logger
from mss import mss
from tesserocr import PSM, PyTessBaseAPI

from utils import Box, Point, Size, imwrite, padding, timer_ns


@define(kw_only=True, eq=False)
class OCR(AbstractContextManager["OCR"]):
    tessdata: Path
    psm: PSM = PSM.AUTO

    tesseract: PyTessBaseAPI = field(init=False)
    _exit_stack: ExitStack = field(init=False)

    CHARACTER_MIN_WIDTH = 60
    CHARACTER_MIN_HEIGHT = 115

    @override
    def __enter__(self) -> Self:
        super().__enter__()

        with ExitStack() as exit_stack:
            self.tesseract = exit_stack.enter_context(PyTessBaseAPI(path=str(self.tessdata), psm=self.psm))
            self._exit_stack = exit_stack.pop_all()
        return self

    def close(self) -> None:
        self._exit_stack.close()

    @override
    def __exit__(
        self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: TracebackType | None
    ) -> None:
        self.close()

    @classmethod
    def preprocess(cls, image: MatLike) -> MatLike:
        """Inspired by https://stackoverflow.com/a/67392752."""

        def get_black_pixels(image: MatLike) -> MatLike:
            """Return a mask of the black pixels in the image."""
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV)[1]

        def find_character_contours(image: MatLike) -> list[MatLike]:
            """Return contours that probably belong to actual characters."""
            return [
                contour
                for contour in cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
                if (bounding_rect := Box(*cv2.boundingRect(contour))).width >= cls.CHARACTER_MIN_WIDTH
                and bounding_rect.height >= cls.CHARACTER_MIN_HEIGHT
            ]

        def mask(image: MatLike, contours: Sequence[MatLike], *, convex_hull: bool = False) -> MatLike:
            """Create a mask from the contours (or their convex hull) and apply it to the image."""
            mask = np.zeros(image.shape[:2], np.uint8)
            if convex_hull:
                hull = cv2.convexHull(np.vstack(contours))
                contours = [hull]
            cv2.drawContours(mask, contours, -1, [255], thickness=-1)
            return cv2.bitwise_and(image, image, mask=mask)

        def gain_division(image: MatLike) -> MatLike:
            # Get local maximum
            kernel_size = 5
            max_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, Size(kernel_size, kernel_size))
            local_max = cv2.morphologyEx(image, cv2.MORPH_CLOSE, max_kernel, borderType=cv2.BORDER_REFLECT101)

            # Perform gain division
            image = np.where(local_max == 0, 0, np.divide(image, local_max, where=local_max != 0))
            # Scale values to [0, 255]
            image = np.multiply(image, 255)
            # Convert mat type from float64 to uint8
            return image.astype(np.uint8)

        def binarize(image: MatLike) -> MatLike:
            """Threshold image using Otsu's method."""
            return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

        def flood_fill(image: MatLike) -> MatLike:
            # Flood fill (white + black)
            cv2.floodFill(image, cast(MatLike, None), Point(0, 0), [255])
            # Invert image so target blobs are colored in white
            return cv2.bitwise_not(image)

        def fill_holes(image: MatLike) -> MatLike:
            # Find the blobs
            contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for contour, current_hierarchy in zip(contours, [] if hierarchy is None else hierarchy[0], strict=True):
                # Look only for children contours (the holes)
                if current_hierarchy[3] != -1:
                    # Use the center of the contour as the flood fill seed point
                    bounding_rect = Box(*cv2.boundingRect(contour))
                    seed_point = bounding_rect.top_left.offset(bounding_rect.width // 2, bounding_rect.height // 2)
                    # Fill the hole
                    cv2.floodFill(image, cast(MatLike, None), seed_point, [0])
            return image

        original_image = image
        imwrite("OCR_0_screenshot.png", image)

        image = gain_division(image)
        imwrite("OCR_1_gain_division.png", image)

        # Convert BGRA to grayscale
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)

        # Resize the image to improve binarization quality
        image = cv2.resize(image, None, fx=3, fy=3)

        image = binarize(image)
        imwrite("OCR_2_binarization.png", image)

        # Get a mask of the black pixels in the original image to find the character borders
        black_borders = get_black_pixels(original_image)
        # Resize the mask to match the image
        black_borders = cv2.resize(black_borders, Size.of(image))
        # Mask the convex hull of the character borders
        image = mask(image, find_character_contours(black_borders), convex_hull=True)
        imwrite("OCR_3_mask_convex_hull.png", image)

        image = flood_fill(image)
        imwrite("OCR_4_flood_fill.png", image)

        image = fill_holes(image)
        imwrite("OCR_5_fill_holes.png", image)

        # Mask the inside of the characters
        image = mask(image, find_character_contours(image))
        imwrite("OCR_6_mask_characters.png", image)

        return image

    def ocr(self, region: Box, *, alphabet: str = "") -> None:
        with mss() as sct:
            image = np.asarray(sct.grab(padding(region)._asdict()), np.uint8)

        with timer_ns() as t:
            preprocessed = self.preprocess(image)
        logger.debug("Preprocessing took {} ms", t() / 1e6)

        if alphabet:
            self.tesseract.SetVariable("tessedit_char_whitelist", alphabet)
        size = Size.of(preprocessed)
        self.tesseract.SetImageBytes(preprocessed.tobytes(), *size, 1, size.width)

        with timer_ns() as t:
            assert self.tesseract.Recognize()

        text: str = self.tesseract.GetUTF8Text()
        logger.debug("OCR found {!r} in {} ms", text.strip(), t() / 1e6)
