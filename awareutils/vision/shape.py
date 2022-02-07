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
    def copy(self) -> "Shape":
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
        if maximum is not None and d > maximum:
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
        return f"Pixel: x={self._x}, y={self._y}"

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

    def copy(self) -> "Pixel":
        return self.__class__(
            x=self.x,
            y=self.y,
            clip=self._clip,
            fix_numeric_type=self._fix_numeric_type,
            isize=self.isize.copy(),
        )

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
        p0: Pixel,
        p1: Pixel,
        clip: bool = False,
        fix_numeric_type: bool = True,
    ):

        if not isinstance(p0, Pixel):
            raise ValueError("p0 should be a Pixel")
        if not isinstance(p1, Pixel):
            raise ValueError("p1 should be a Pixel")
        if p0.isize != p1.isize:
            raise ValueError("p0 and p1 should have the same isize i.e. coordinate system.")
        if p0.x > p1.x:
            raise ValueError("p0.x should be <= p1.x")
        if p0.y > p1.y:
            raise ValueError("p0.y should be <= p1.y")

        self._p0 = p0
        self._p1 = p1

        super().__init__(isize=p0.isize, clip=clip, fix_numeric_type=fix_numeric_type)

    @staticmethod
    def from_x0y0x1y1(
        isize: ImgSize, x0: int, y0: int, x1: int, y1: int, clip: bool = False, fix_numeric_type: bool = True
    ):
        p0 = Pixel(isize=isize, x=x0, y=y0)
        p1 = Pixel(isize=isize, x=x1, y=y1)
        return Rectangle(p0=p0, p1=p1, clip=clip, fix_numeric_type=fix_numeric_type)

    @staticmethod
    def from_x0y0wh(
        x0: int, y0: int, w: int, h: int, isize: ImgSize, clip: bool = False, fix_numeric_type: bool = True
    ) -> "Rectangle":
        # Note the -1 here as x1/y1 are inclusive. If the box has width 1, i.e. a point, then x1 should equal x0.
        p0 = Pixel(isize=isize, x=x0, y=y0)
        p1 = Pixel(isize=isize, x=x0 + w - 1, y=y0 + h - 1)
        return Rectangle(p0=p0, p1=p1, clip=clip, fix_numeric_type=fix_numeric_type)

    @property
    def p0(self) -> Pixel:
        return self._p0

    @property
    def p1(self) -> Pixel:
        return self._p1

    @property
    def x0(self) -> int:
        return self._p0.x

    @property
    def y0(self) -> int:
        return self._p0.y

    @property
    def x1(self) -> int:
        return self._p1.x

    @property
    def y1(self) -> int:
        return self._p1.y

    def __repr__(self):
        return f"Rectangle: x0={self.x0}, x1={self.x1}, y0={self.y0}, y1={self.y1}"

    @property
    def center(self) -> "Pixel":
        return Pixel(
            x=int((self.x1 + self.x0) / 2),
            y=int((self.y1 + self.y0) / 2),
            isize=self._isize,
            clip=self._clip,
            fix_numeric_type=self._fix_numeric_type,
        )

    @property
    def w(self) -> int:
        return self.x1 - self.x0 + 1  # Since our x1 is inclusive

    @property
    def h(self) -> int:
        return self.y1 - self.y0 + 1  # Since our y1 is inclusize

    @property
    def area(self) -> int:
        return self.h * self.w

    def copy(self) -> "Rectangle":
        return self.__class__(
            p0=self.p0.copy(),
            p1=self.p1.copy(),
            clip=self._clip,
            fix_numeric_type=self._fix_numeric_type,
            isize=self.isize.copy(),
        )

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
        return arr[self.y0 : self.y1 + 1, self.x0 : self.x1 + 1, ...]

    def intersection(self, rect: "Rectangle") -> Optional["Rectangle"]:
        if self._isize != rect.isize:
            raise RuntimeError("Can't compute IOU for rectangles defined one different image sizes.")
        x0 = max(self.x0, rect.x0)
        y0 = max(self.y0, rect.y0)
        x1 = min(self.x1, rect.x1)
        y1 = min(self.y1, rect.y1)
        if x0 >= x1 or y0 >= y1:
            return None
        else:
            return Rectangle.from_x0y0x1y1(x0=x0, y0=y0, x1=x1, y1=y1, isize=self._isize)

    def iou(self, rect: "Rectangle") -> float:
        intersection = self.intersection(rect)
        if intersection is None:
            return 0
        return intersection.area / (self.area + rect.area - intersection.area)

    def _project(self, isize: ImgSize) -> "Rectangle":
        return Rectangle(
            p0=self._p0.project(isize=isize),
            p1=self._p1.project(isize=isize),
            clip=self._clip,
            fix_numeric_type=self._fix_numeric_type,
        )


class PolyLine(Shape):
    def __init__(self, *, pixels: List[Pixel], clip: bool = False, fix_numeric_type: bool = True):

        if not isinstance(pixels, (list, tuple)):
            raise RuntimeError("pixels should be a list of tuple")
        if len(pixels) == 0:
            raise RuntimeError("must be at least one pixel to define a polygon!")
        if not all(isinstance(p, Pixel) for p in pixels):
            raise ValueError("All pixels must be Pixels")
        isizes = set([p.isize for p in pixels])
        if len(isizes) != 1:
            raise ValueError(f"All pixels should have the same isize but there are {len(isizes)}.")
        super().__init__(isize=isizes.pop(), clip=clip, fix_numeric_type=fix_numeric_type)
        self._pixels = list(pixels)

    def __repr__(self) -> str:
        return f"PolyLine: pixels={self.pixels}"

    @staticmethod
    def from_xy(isize: ImgSize, xy: List, clip: bool = False, fix_numeric_type: bool = True) -> "PolyLine":
        if not isinstance(xy, (tuple, list)):
            raise RuntimeError("xy should be a tuple or list")
        if len(xy) < 2:
            raise RuntimeError("xy should have at least two elements")
        if not all(isinstance(i, (tuple, list)) for i in xy):
            raise RuntimeError("xy should be a tuple or list of tuples or lists")
        pixels = [Pixel(isize=isize, x=x, y=y) for x, y in xy]
        return PolyLine(pixels=pixels, clip=clip, fix_numeric_type=fix_numeric_type)

    @property
    def pixels(self) -> List[Pixel]:
        return self._pixels

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

    def copy(self) -> "PolyLine":
        return self.__class__(
            pixels=[p.copy() for p in self.pixels],
            clip=self._clip,
            fix_numeric_type=self._fix_numeric_type,
            isize=self.isize.copy(),
        )

    def _project(self, isize: ImgSize) -> "PolyLine":
        return PolyLine(
            isize=isize,
            pixels=[pixel.project(isize) for pixel in self._pixels],
            clip=self._clip,
            fix_numeric_type=self._fix_numeric_type,
        )


class Line(PolyLine):
    def __init__(self, *, p0: Pixel, p1: Pixel, clip: bool = False, fix_numeric_type: bool = True):

        if not isinstance(p0, Pixel):
            raise ValueError("p0 should be a Pixel")
        if not isinstance(p1, Pixel):
            raise ValueError("p1 should be a Pixel")
        super().__init__(pixels=(p0, p1), clip=clip, fix_numeric_type=fix_numeric_type)

    @staticmethod
    def from_xy(isize: ImgSize, xy: List, clip: bool = False, fix_numeric_type: bool = True) -> "Line":
        polyline = super().from_xy(isize=isize, xy=xy, clip=clip, fix_numeric_type=fix_numeric_type)
        return Line(p0=polyline.pixels[0], p1=polyline.pixels[1], clip=clip, fix_numeric_type=fix_numeric_type)

    @property
    def p0(self) -> Pixel:
        return self._pixels[0]

    @property
    def p1(self) -> Pixel:
        return self._pixels[1]

    def copy(self) -> "Line":
        return self.__class__(
            p0=self.p0.copy(),
            p1=self.p1.copy(),
            clip=self._clip,
            fix_numeric_type=self._fix_numeric_type,
            isize=self.isize.copy(),
        )

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
    def __init__(self, *, pixels: List[Pixel], clip: bool = False, fix_numeric_type: bool = True):
        # Check polygon is not closed
        if pixels[0] == pixels[-1]:
            logger.info("Polygons are specified without closure - ignoring the last point")
            pixels = pixels[:-1]
        super().__init__(pixels=pixels, clip=clip, fix_numeric_type=fix_numeric_type)

    @staticmethod
    def from_xy(isize: ImgSize, xy: List, clip: bool = False, fix_numeric_type: bool = True) -> "Polygon":
        polyline = super().from_xy(isize=isize, xy=xy, clip=clip, fix_numeric_type=fix_numeric_type)
        return Polygon(pixels=polyline.pixels, clip=clip, fix_numeric_type=fix_numeric_type)

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
    def __init__(self, *, center: Pixel, radius: int, clip: bool = False, fix_numeric_type: bool = True):
        # TODO: check if circle is within the img or not?
        if not isinstance(center, Pixel):
            raise RuntimeError("center should be a Pixel")
        if not (isinstance(radius, int) and radius > 0):
            raise RuntimeError("radius should be a positive int")
        super().__init__(isize=center.isize, clip=clip, fix_numeric_type=fix_numeric_type)
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

    def copy(self) -> "Circle":
        return self.__class__(
            center=self.center.copy(),
            radius=self.radius,
            clip=self._clip,
            fix_numeric_type=self._fix_numeric_type,
            isize=self.isize.copy(),
        )

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
