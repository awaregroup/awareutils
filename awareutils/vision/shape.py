import math
from abc import ABCMeta, abstractmethod, abstractproperty
from typing import List, Optional

import numpy as np
from awareutils.vision.img import ImgSize
from loguru import logger


class Shape(metaclass=ABCMeta):
    def __init__(self, *, isize: ImgSize, clip: bool = False, fix_numeric_type: bool = True):
        self._isize = isize
        self._clip = clip
        self._fix_numeric_type = fix_numeric_type

    @property
    def isize(self) -> ImgSize:
        return self._isize

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
    def _project(self, isize: ImgSize):
        """
        Project - where you can assume the sizes are different.
        """
        pass

    def project(self, isize: ImgSize):
        if self._isize == isize:
            return self
        return self._project(isize)

    def _project_x(self, x: int, isize: ImgSize):
        return int(x / self._isize.w * isize.w)

    def _project_y(self, y: int, isize: ImgSize):
        return int(y / self._isize.h * isize.h)

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
        return self._validate_coordinate(d=x, maximum=self._isize.w - 1)

    def _validate_y(self, y: int) -> int:
        return self._validate_coordinate(d=y, maximum=self._isize.h - 1)


class Pixel(Shape):
    def __init__(self, *, x: int, y: int, isize: ImgSize, clip: bool = False, fix_numeric_type: bool = True):
        super().__init__(isize=isize, clip=clip, fix_numeric_type=fix_numeric_type)
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

    def _project(self, isize: ImgSize) -> "Pixel":
        return Pixel(
            isize=isize,
            x=self._project_x(self._x, isize=isize),
            y=self._project_y(self._y, isize=isize),
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
        isize: ImgSize,
        clip: bool = False,
        fix_numeric_type: bool = True,
    ):

        super().__init__(isize=isize, clip=clip, fix_numeric_type=fix_numeric_type)
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
            isize=self._isize,
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
        if h != self._isize.h or w != self._isize.w:
            raise RuntimeError(
                (
                    "This array has a different size to the img this shape was defined with i.e. the coordinate systems"
                    " don't match. Consider using shape.project() first."
                )
            )
        # +1 as our x1/y1 are inclusive, not exclusive like numpy
        return arr[self._y0 : self._y1 + 1, self._x0 : self._x1 + 1, ...]

    def intersection(self, rect: "Rectangle") -> Optional["Rectangle"]:
        if self._isize != rect.isize:
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

    def _project(self, isize: ImgSize) -> "Rectangle":
        return Rectangle(
            isize=isize,
            x0=self._project_x(x=self._x0, isize=isize),
            y0=self._project_y(y=self._y0, isize=isize),
            x1=self._project_x(x=self._x1, isize=isize),
            y1=self._project_y(y=self._y1, isize=isize),
            clip=self._clip,
            fix_numeric_type=self._fix_numeric_type,
        )

    @classmethod
    def from_x0y0wh(
        cls, x0: int, y0: int, w: int, h: int, isize: ImgSize, clip: bool = False, fix_numeric_type: bool = True
    ):
        # Note the -1 here as x1/y1 are inclusive. If the box has width 1, i.e. a point, then x1 should equal x0.
        return cls(
            x0=x0, y0=y0, x1=x0 + w - 1, y1=y0 + h - 1, isize=isize, clip=clip, fix_numeric_type=fix_numeric_type
        )


class PolyLine(Shape):
    def __init__(self, *, pixels: List[Pixel], isize: ImgSize, clip: bool = False, fix_numeric_type: bool = True):

        super().__init__(isize=isize, clip=clip, fix_numeric_type=fix_numeric_type)
        if not isinstance(pixels, (list, tuple)):
            raise RuntimeError("pixels should be a list of tuple")
        if len(pixels) == 0:
            raise RuntimeError("must be at least one pixel to define a polygon!")
        if not all(isinstance(p, Pixel) for p in pixels):
            raise ValueError("All pixels must be Pixels")
        if any(p.isize != isize for p in pixels):
            raise ValueError("All pixels should have isize matching that provided.")
        self._pixels = list(pixels)

    @property
    def pixels(self) -> List[Pixel]:
        return self._pixels

    def __repr__(self):
        return f"PolyLine: pixels={self._pixels}"

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

    def _project(self, isize: ImgSize) -> "PolyLine":
        return PolyLine(
            isize=isize,
            pixels=[pixel.project(isize) for pixel in self._pixels],
            clip=self._clip,
            fix_numeric_type=self._fix_numeric_type,
        )


class Line(PolyLine):
    def __init__(self, *, p0: Pixel, p1: Pixel, isize: ImgSize, clip: bool = False, fix_numeric_type: bool = True):

        if not isinstance(p0, Pixel):
            raise ValueError("p0 should be a Pixel")
        if not isinstance(p1, Pixel):
            raise ValueError("p1 should be a Pixel")
        super().__init__(pixels=(p0, p1), isize=isize, clip=clip, fix_numeric_type=fix_numeric_type)

    @property
    def p0(self) -> Pixel:
        return self._pixels[0]

    @property
    def p1(self) -> Pixel:
        return self._pixels[1]

    def __repr__(self):
        return f"Line: {self.p0} -> {self.p1}"

    def _project(self, isize: ImgSize) -> "Line":
        polyline = super().project(isize)
        return Line(
            isize=isize,
            p0=polyline.pixels[0],
            p1=polyline.pixels[1],
            clip=self._clip,
            fix_numeric_type=self._fix_numeric_type,
        )


class Polygon(PolyLine):
    def __init__(self, *, pixels: List[Pixel], isize: ImgSize, clip: bool = False, fix_numeric_type: bool = True):
        # Check polygon is not closed
        if pixels[0] == pixels[-1]:
            logger.info("Polygons are specified without closure - ignoring the last point")
            pixels = pixels[:-1]
        super().__init__(pixels=pixels, isize=isize, clip=clip, fix_numeric_type=fix_numeric_type)

    @property
    def pixels_closed(self) -> List[Pixel]:
        return self._pixels if len(self._pixels) <= 1 else self._pixels + [self._pixels[-1]]

    def __repr__(self):
        return f"Polygon: points={self._pixels}"

    def _project(self, isize: ImgSize) -> "Polygon":
        polyline = super().project(isize)
        return Polygon(
            isize=isize,
            pixels=polyline.pixels,
            clip=self._clip,
            fix_numeric_type=self._fix_numeric_type,
        )


class Circle(Shape):
    def __init__(
        self, *, isize: ImgSize, center: Pixel, radius: int, clip: bool = False, fix_numeric_type: bool = True
    ):
        # TODO: check if circle is within the img or not?
        super().__init__(isize=isize, clip=clip, fix_numeric_type=fix_numeric_type)
        if not isinstance(center, Pixel):
            raise RuntimeError("center should be a Pixel")
        if not center.isize != isize:
            raise RuntimeError("center.isize != isize i.e. different coordinate systems!")
        if not (isinstance(radius, int) and radius > 0):
            raise RuntimeError("radius should be a positive int")
        self._center = center
        self._radius = radius

    def __repr__(self):
        return f"Cirlce: center={self._center}, radius={self._radius}"

    @property
    def center(self) -> "Pixel":
        return self._center

    @property
    def radius(self) -> int:
        return self._radius

    @property
    def w(self) -> int:
        return self._radius * 2

    @property
    def h(self) -> int:
        return self._radius * 2

    @property
    def area(self) -> int:
        return math.pi * self._radius ** 2

    def _project(self, isize: ImgSize) -> "Circle":
        current_aspect_ratio = self._isize.w / self._isize.h
        new_aspect_ratio = isize.w / isize.h
        if current_aspect_ratio != new_aspect_ratio:
            raise RuntimeError(
                (
                    "Reprojecting circles isn't defined if aspect ratio of the img change, as what do we scale the "
                    "radius by - the increased height or width ratio (which don't match)? Alternatively, do we do both "
                    "and end up with an ellipse?"
                )
            )
        return Circle(
            isize=isize,
            center=self._center.project(isize=isize),
            radius=int(self._radius / self._isize.w * isize.w),
            clip=self._clip,
            fix_numeric_type=self._fix_numeric_type,
        )
