from abc import ABCMeta, abstractmethod

import cv2
import numpy as np
from awareutils.vision.col import Col
from awareutils.vision.img import Img, ImgType
from awareutils.vision.shape import Pixel, Polygon, Rectangle, Shape
from loguru import logger
from PIL import ImageDraw


def _none_or_rgb(col: Col):
    return None if col is None else col.rgb


class Drawer(metaclass=ABCMeta):
    def __init__(self, img: Img, reproject_shapes_if_required: bool = True):
        if img is None:
            raise RuntimeError("img must not be None")
        if not isinstance(img, Img):
            raise RuntimeError("img must be an Img")
        self.img = img
        self._reproject_shapes_if_required = reproject_shapes_if_required

    # Drawing:
    def _check_args(self, shape: Shape, outline: Col = None, fill: Col = None, width: int = 1) -> Shape:
        if not isinstance(shape, Shape):
            raise ValueError("'shape' must be a Shape")
        if outline is None and fill is None:
            raise ValueError("Please specify a color for at least one of `fill` or `outline`")
        if outline is not None and not isinstance(outline, Col):
            raise ValueError("outline should be a col")
        if fill is not None and not isinstance(fill, Col):
            raise ValueError("fill should be a Col")
        if not isinstance(width, int):
            raise ValueError("width should be an int")

        # Re-project if needed
        if self.img.size != shape._img_size:
            if self._reproject_shapes_if_required:
                logger.warning("Img and shape size don't match, so reprojecting shape to img size before drawing.")
                shape = shape.project(self.size)
            else:
                raise RuntimeError(
                    (
                        "This img has a different size to the img thes shape was defined with i.e. the coordinate"
                        " systems don't match. Consider using shape.project() first."
                    )
                )
        return shape

    @abstractmethod
    def draw_pixel(self, pixel: Pixel, fill: Col = None, outline: Col = None, width: int = 1) -> None:
        pass

    @abstractmethod
    def draw_rectangle(self, rectangle: Rectangle, fill: Col = None, outline: Col = None, width: int = 1) -> None:
        pass

    @abstractmethod
    def draw_polygon(self, polygon: Polygon, fill: Col = None, outline: Col = None, width: int = 1) -> None:
        pass

    def draw(self, shape: Shape, fill: Col = None, outline: Col = None, width: int = 1) -> None:
        method = getattr(self, f"draw_{shape.__class__.__name__.lower()}")
        return method(shape, fill=fill, outline=outline, width=width)


class PILDrawer(Drawer):
    def __init__(self, img: Img, reproject_shapes_if_required: bool = True):
        super().__init__(img=img, reproject_shapes_if_required=reproject_shapes_if_required)
        if img.itype != ImgType.PIL:
            raise RuntimeError("img must be a PIL Img")
        self._imgdraw = ImageDraw.Draw(img.source)

    def draw_pixel(self, pixel: Pixel, fill: Col = None, outline: Col = None, width: int = 1) -> None:
        pixel = self._check_args(shape=pixel, outline=outline, fill=fill, width=width)
        if fill is None:
            raise ValueError("Please provide a `fill` to draw a Pixel")
        self._imgdraw.point(xy=(pixel.x, pixel.y), fill=fill.rgb)

    def draw_rectangle(self, rectangle: Rectangle, fill: Col = None, outline: Col = None, width: int = 1) -> None:
        rectangle = self._check_args(shape=rectangle, outline=outline, fill=fill, width=width)
        self._imgdraw.rectangle(
            (rectangle.x0, rectangle.y0, rectangle.x1, rectangle.y1),
            fill=_none_or_rgb(fill),
            outline=_none_or_rgb(outline),
            width=width,
        )

    def draw_polygon(self, polygon: Polygon, fill: Col = None, outline: Col = None, width: int = 1) -> None:
        polygon = self._check_args(shape=polygon, outline=outline, fill=fill, width=width)
        # ImageDraw.polygon doesn't support width, but line does. So use polygon unless we need an outline of width != 1
        points = [(p.x, p.y) for p in polygon.pixels]
        self._imgdraw.polygon(points, fill=_none_or_rgb(fill.rgb), outline=_none_or_rgb(outline.rgb))
        # OK, the cases where we need custom width ...
        if outline is not None and width != 1 and (fill is None or fill != outline):
            points = [(p.x, p.y) for p in polygon.pixels_closed]
            self._imgdraw.line(points, outline=outline.rgb, width=width)


class OpenCVDrawer(Drawer):
    def __init__(self, img: Img, reproject_shapes_if_required: bool = True):
        super().__init__(img=img, reproject_shapes_if_required=reproject_shapes_if_required)
        if img.itype not in (ImgType.BGR, ImgType.RGB):
            raise RuntimeError("img must be a BGR or RGB img")
        self._contiguity_test()

    def _contiguity_test(self):
        if not self.img.source.flags.c_contiguous:
            raise RuntimeError("Source array isn't contiguous, which breaks OpenCV drawing. This shouldn't happen ...")

    def draw_pixel(self, pixel: Pixel, fill: Col = None, outline: Col = None, width: int = 1) -> None:
        self._contiguity_test()
        pixel = self._check_args(shape=pixel, outline=outline, fill=fill, width=width)
        if fill is None:
            raise ValueError("Please provide a `fill` to draw a Pixel")
        self.img.source[pixel.y, pixel.x, :] = self._col(fill)

    def draw_rectangle(self, rectangle: Rectangle, fill: Col = None, outline: Col = None, width: int = 1) -> None:
        self._contiguity_test()
        rectangle = self._check_args(shape=rectangle, outline=outline, fill=fill, width=width)
        # OK, OpenCV doesn't let us do fill and line separately, so let's do both.
        p0 = (rectangle.x0, rectangle.y0)
        p1 = (rectangle.x1, rectangle.y1)
        # Always fill first:
        if fill is not None:
            cv2.rectangle(self.img.source, p0, p1, color=self._col(fill), thickness=-1)
        # Only outline if we need to:
        if outline is not None and (fill is None or outline != fill or width > 1):
            cv2.rectangle(self.img.source, p0, p1, color=self._col(outline), thickness=width)

    def draw_polygon(self, polygon: Polygon, fill: Col = None, outline: Col = None, width: int = 1) -> None:
        self._contiguity_test()
        polygon = self._check_args(shape=polygon, outline=outline, fill=fill, width=width)
        # OK, OpenCV doesn't let us do fill and line separately, so let's do both.
        pts = [np.array([(p.x, p.y) for p in polygon.pixels])]
        # Always fill first:
        if fill is not None:
            cv2.fillPoly(self.img.source, pts=pts, color=self._col(fill))
        # Only outline if we need to:
        if outline is not None and (fill is None or outline != fill or width > 1):
            cv2.polylines(self.img.source, pts=pts, closed=True, color=self._col(outline), thickness=width)

    def _col(self, col: Col):
        if col is None:
            return None
        return col.rgb if self.img.itype == ImgType.RGB else col.bgr
