from abc import ABCMeta, abstractmethod, abstractproperty
from typing import List, Optional

import numpy as np
from awareutils.vision.img import Img
from awareutils.vision.img_size import ImgSize
from loguru import logger


def img_operation(f):
    """
    Wrapper to pass when the first argument is the img, and the rest are whatever. I.e. signature should be 
        f(self, img: Img, *, ...)
    Basically just checks the img.size attribute matches our size.
    """

    def wrapper(*args, **kwargs):
        self = args[0]
        img = args[1]
        if not isinstance(img, Img):
            raise ValueError("'img' must be an Img")
        if img.size != self._img_size:
            raise RuntimeError(
                (
                    "This img has a different size to the img this shape was defined with i.e. the coordinate systems"
                    " don't match. Consider using shape.project() first."
                )
            )
        return f(*args, **kwargs)

    return wrapper


class Shape(metaclass=ABCMeta):
    def __init__(self, *, img_size: ImgSize, clip: bool = False, fix_numeric_type: bool = True):
        self._img_size = img_size
        self._clip = clip
        self._fix_numeric_type = fix_numeric_type

    @abstractproperty
    def center(self) -> "Pixel":
        pass

    @abstractproperty
    def w(self) -> int:
        pass

    @abstractproperty
    def h(self) -> int:
        pass

    @abstractproperty
    def area(self) -> int:
        pass

    @abstractmethod
    def _project(self, img_size: ImgSize):
        """
        Project - where you can assume the sizes are different.
        """
        pass

    def project(self, img_size: ImgSize):
        if self._img_size == img_size:
            return self
        return self._project(img_size)

    def _project_x(self, x: int, img_size: ImgSize):
        return int(x / self._img_size.w * img_size.w)

    def _project_y(self, y: int, img_size: ImgSize):
        return int(y / self._img_size.h * img_size.h)

    def _validate_coordinate(self, d: int, maximum: int = None) -> int:
        if d is None:
            raise ValueError("Coordinate must not be None")

        if not isinstance(d, int):
            if self._fix_numeric_type:
                logger.debug("Coordinate is meant to be pixel and hence an int, so trying to coerce to int")
                d = int(d)
            else:
                raise ValueError("Coordinate is meant to be pixel but isn't an int")

        # Should always be >= 0
        if d < 0:
            if self._clip:
                d = 0
                logger.debug("Coordinate must be >= 0 so clipping to 0")
            else:
                raise ValueError("Coordinate must be >= 0")

        # And *less than* (but not equal to!) img coordinates:
        if maximum is not None and d >= maximum:
            if self._clip:
                d = maximum
                logger.debug("Coordinate must be < {maximum} - clipping.", maximum=maximum)
            else:
                raise ValueError("Coordinate must be < maximum")

        # Phew, done:
        return d

    def _validate_x(self, x: int) -> int:
        return self._validate_coordinate(d=x, maximum=self._img_size.w - 1)

    def _validate_y(self, y: int) -> int:
        return self._validate_coordinate(d=y, maximum=self._img_size.h - 1)

    # @img_operation
    # @abstractmethod
    # def fill(self, img: Img, *, color) -> None:
    #     pass

    # @img_operation
    # @abstractmethod
    # def outline(self, img: Img) -> None:
    #     pass


class Pixel(Shape):
    def __init__(self, *, x: int, y: int, img_size: ImgSize, clip: bool = False, fix_numeric_type: bool = True):
        super().__init__(img_size=img_size, clip=clip, fix_numeric_type=fix_numeric_type)
        self.x = x
        self.y = y

    @property
    def x(self) -> int:
        return self._x

    @x.setter
    def x(self, x) -> None:
        self._x = self._validate_x(x)

    @property
    def y(self) -> int:
        return self._y

    @y.setter
    def y(self, y) -> None:
        self._y = self._validate_y(y)

    def __repr__(self):
        return f"Point: x={self._x}, y={self._y}"

    @property
    def center(self) -> "Pixel":
        return self

    @property
    def w(self) -> int:
        return 1

    @property
    def h(self) -> int:
        return 1

    @property
    def area(self) -> int:
        return 1

    def _project(self, img_size: ImgSize) -> "Pixel":
        return Pixel(
            img_size=img_size,
            x=self._project_x(self._x, img_size=img_size),
            y=self._project_y(self._y, img_size=img_size),
            clip=self._clip,
            fix_numeric_type=self._fix_numeric_type,
        )


class Rectangle(Shape):
    def __init__(
        self,
        *,
        x0: int,
        x1: int,
        y0: int,
        y1: int,
        img_size: ImgSize,
        clip: bool = False,
        fix_numeric_type: bool = True,
    ):

        super().__init__(img_size=img_size, clip=clip, fix_numeric_type=fix_numeric_type)
        self._x0 = self._validate_x(x0)
        self._y0 = self._validate_y(y0)
        self._x1 = self._validate_x(x1)
        self._y1 = self._validate_y(y1)
        self._validate_box()

    @property
    def x0(self) -> int:
        return self._x0

    @property
    def y0(self) -> int:
        return self._y0

    @property
    def x1(self) -> int:
        return self._x1

    @property
    def y1(self) -> int:
        return self._y1

    def __repr__(self):
        return f"Box: x0={self._x0}, x1={self._x1}, y0={self._y0}, y1={self._y1})"

    def _validate_box(self) -> None:
        if self._x0 > self._x1:
            raise ValueError("x0 should be <= x1")
        if self._y0 > self._y1:
            raise ValueError("y0 should be <= y1")

    @property
    def center(self) -> "Pixel":
        return Pixel(
            x=int((self._x1 + self._x0) / 2),
            y=int((self._y1 + self._y0) / 2),
            img_size=self._img_size,
            clip=self._clip,
            fix_numeric_type=self._fix_numeric_type,
        )

    @property
    def w(self) -> int:
        return self._x1 - self._x0 + 1  # Since our x1 is inclusive

    @property
    def h(self) -> int:
        return self._y1 - self._y0 + 1  # Since our y1 is inclusize

    @property
    def area(self) -> int:
        return self.h * self.w

    def slice_array(self, arr: np.ndarray) -> np.ndarray:
        # TODO: this assumes mono or color array and not e.g. stacks of colors
        # TODO: maybe allow users to slice without providing arr? It's only used for h/w checking so will work without
        #   it ... but removing the checks seems like a bad idea for slightly shorter code.
        if not isinstance(arr, np.ndarray):
            raise RuntimeError("Can only slice numpy arrays")
        h, w = arr.shape[:2]
        if h != self._img_size.h or w != self._img_size.w:
            raise RuntimeError(
                (
                    "This array has a different size to the img this shape was defined with i.e. the coordinate systems"
                    " don't match. Consider using shape.project() first."
                )
            )
        # +1 as our x1/y1 are inclusive, not exclusive like numpy
        return arr[self._y0 : self._y1 + 1, self._x0 : self._x1 + 1, ...]

    def intersection(self, rect: "Rectangle") -> Optional["Rectangle"]:
        if self._img_size != rect._img_size:
            raise RuntimeError("Can't compute IOU for rectangles defined one different image sizes.")
        x0 = max(self._x0, rect.x0)
        y0 = max(self._y0, rect.y0)
        x1 = min(self._x1, rect.x1)
        y1 = min(self._y1, rect.y1)
        if x0 >= x1 or y0 >= y1:
            return None
        else:
            return self.__class__(x0=x0, y0=y0, x1=x1, y1=y1)

    def iou(self, rect: "Rectangle") -> float:
        intersection = self.intersection(rect)
        if intersection is None:
            return 0
        return intersection.area / (self.area + rect.area - intersection.area)

    def _project(self, img_size: ImgSize) -> "Rectangle":
        return Rectangle(
            img_size=img_size,
            x0=self._project_x(x=self._x0, img_size=img_size),
            y0=self._project_y(y=self._y0, img_size=img_size),
            x1=self._project_x(x=self._x1, img_size=img_size),
            y1=self._project_y(y=self._y1, img_size=img_size),
            clip=self._clip,
            fix_numeric_type=self._fix_numeric_type,
        )

    @classmethod
    def from_x0y0wh(
        cls, x0: int, y0: int, w: int, h: int, img_size: ImgSize, clip: bool = False, fix_numeric_type: bool = True
    ):
        # Note the -1 here as x1/y1 are inclusive. If the box has width 1, i.e. a point, then x1 should equal x0.
        return cls(
            x0=x0, y0=y0, x1=x0 + w - 1, y1=y0 + h - 1, img_size=img_size, clip=clip, fix_numeric_type=fix_numeric_type
        )


class Polygon(Shape):
    def __init__(self, *, pixels: List[Pixel], img_size: ImgSize, clip: bool = False, fix_numeric_type: bool = True):

        super().__init__(img_size=img_size, clip=clip, fix_numeric_type=fix_numeric_type)
        if not isinstance(pixels, (list, tuple)):
            raise RuntimeError("pixels should be a list of tuple")
        if len(pixels) == 0:
            raise RuntimeError("must be at least one pixel to define a polygon!")
        if not all(isinstance(p, Pixel) for p in pixels):
            raise ValueError("All points in pixels must be Pixels")
        self._pixels = pixels

    @property
    def pixels(self) -> List[Pixel]:
        return self._pixels

    def __repr__(self):
        return f"Polygon: points={self._pixels}"

    @property
    def center(self) -> "Pixel":
        raise NotImplementedError("TODO")

    @property
    def w(self) -> int:
        raise NotImplementedError("TODO")

    @property
    def h(self) -> int:
        raise NotImplementedError("TODO")

    @property
    def area(self) -> int:
        raise NotImplementedError("TODO")

    def _project(self, img_size: ImgSize) -> "Rectangle":
        return Polygon(
            img_size=img_size,
            pixels=[pixel._project(img_size) for pixel in self._pixels],
            clip=self._clip,
            fix_numeric_type=self._fix_numeric_type,
        )
