from pathlib import Path

import numpy as np
import pytest
from awareutils.vision.col import Col
from awareutils.vision.img import Img, ImgSize, ImgType
from awareutils.vision.shape import Circle, Line, Pixel, Polygon, PolyLine, Rectangle

DRAW_DIR = Path(__file__).resolve().parent / "draws"
ISIZE = ImgSize(h=50, w=50)
PIXEL = Pixel(x=10, y=10, isize=ISIZE)
PIXEL2 = Pixel(x=20, y=20, isize=ISIZE)
PIXEL3 = Pixel(x=5, y=45, isize=ISIZE)
CIRCLE = Circle(center=PIXEL2, radius=15)
POLYLINE = PolyLine(pixels=[PIXEL, PIXEL2, PIXEL3])
POLYGON = Polygon(pixels=[PIXEL, PIXEL2, PIXEL3])
RECT = Rectangle(p0=PIXEL, p1=PIXEL2)
LINE = Line(p0=PIXEL, p1=PIXEL2)
COL = Col.named.aware_blue_dark
COL2 = Col.named.aware_blue_light

xfail = pytest.mark.xfail

"""
NB: try to test drawings are correct algorithmically if it's easy and clear - otherwise compare to known "correct" imgs.
"""


@pytest.mark.parametrize("itype", (ImgType.PIL, ImgType.BGR, ImgType.RGB))
def test_draw_pixel(itype: ImgType):
    img = Img.new(ISIZE, itype)
    img.draw.draw(PIXEL, fill=COL)
    rgb = img.rgb()
    assert np.array_equal(rgb[10, 10, :], COL.rgb)


def _compared_to_saved(img: Img, name: str, differ: bool, save: bool = False):
    if differ:
        drawtype = "pil" if img.itype == ImgType.PIL else "cv"
        expected = DRAW_DIR / f"{name}_{drawtype}.png"
    else:
        expected = DRAW_DIR / f"{name}.png"
    if save:
        img.save(str(expected)[:-4] + ".save.png")
    expected = Img.open_rgb(expected)
    assert np.array_equal(img.rgb(), expected.rgb())


@pytest.mark.parametrize("itype", (ImgType.PIL, ImgType.BGR, ImgType.RGB))
def test_draw_line(itype: ImgType):
    img = Img.new(ISIZE, itype)
    img.draw.draw(LINE, outline=COL)
    rgb = img.rgb()
    for i in range(10, 21):
        assert np.array_equal(rgb[i, i, :], COL.rgb)


@pytest.mark.parametrize("itype", (ImgType.PIL, ImgType.BGR, ImgType.RGB))
def test_draw_line_thick(itype: ImgType):
    img = Img.new(ISIZE, itype)
    img.draw.draw(LINE, outline=COL, width=2)
    _compared_to_saved(img, "line_thick", differ=True)


@pytest.mark.parametrize("itype", (ImgType.PIL, ImgType.BGR, ImgType.RGB))
def test_draw_polyline(itype: ImgType):
    img = Img.new(ISIZE, itype)
    img.draw.draw(POLYLINE, outline=COL)
    _compared_to_saved(img, "polyline", differ=False)


@pytest.mark.parametrize("itype", (ImgType.PIL, ImgType.BGR, ImgType.RGB))
def test_draw_polyline_thick(itype: ImgType):
    img = Img.new(ISIZE, itype)
    img.draw.draw(POLYLINE, outline=COL, width=2)
    _compared_to_saved(img, "polyline_thick", differ=True)


@pytest.mark.parametrize("itype", (ImgType.PIL, ImgType.BGR, ImgType.RGB))
def test_draw_rectangle_filled(itype: ImgType):
    img = Img.new(ISIZE, itype)
    img.draw.draw(RECT, fill=Col(1, 2, 3))
    rgb = img.rgb()
    assert np.all(rgb[10:21, 10:21, 0] == 1)
    assert np.all(rgb[10:21, 10:21, 1] == 2)
    assert np.all(rgb[10:21, 10:21, 2] == 3)


@pytest.mark.parametrize("itype", (ImgType.PIL, ImgType.BGR, ImgType.RGB))
def test_draw_rectangle_outlined(itype: ImgType):
    img = Img.new(ISIZE, itype)
    img.draw.draw(RECT, outline=Col(1, 1, 1), width=1)
    rgb = img.rgb()
    assert rgb.sum() == (11 * 2 + 9 * 2) * 3


@pytest.mark.parametrize("itype", (ImgType.PIL, ImgType.BGR, ImgType.RGB))
def test_draw_rectangle_outlined_thick(itype: ImgType):
    img = Img.new(ISIZE, itype)
    img.draw.draw(RECT, outline=COL, width=2)
    _compared_to_saved(img, "rectangle_outline_thick", differ=True)


@pytest.mark.parametrize("itype", (ImgType.PIL, ImgType.BGR, ImgType.RGB))
def test_draw_rectangle_filled_outlined_thick(itype: ImgType):
    img = Img.new(ISIZE, itype)
    img.draw.draw(RECT, outline=COL, width=2, fill=COL2)
    _compared_to_saved(img, "rectangle_filled_outline_thick", differ=True)


@pytest.mark.parametrize("itype", (ImgType.PIL, ImgType.BGR, ImgType.RGB))
def test_draw_circle_outlined(itype: ImgType):
    img = Img.new(ISIZE, itype)
    img.draw.circle(CIRCLE, outline=COL, width=1)
    _compared_to_saved(img, "circle_outline", differ=True)


@pytest.mark.parametrize("itype", (ImgType.PIL, ImgType.BGR, ImgType.RGB))
def test_draw_circle_outlined_thick(itype: ImgType):
    img = Img.new(ISIZE, itype)
    img.draw.draw(CIRCLE, outline=COL, width=2)
    _compared_to_saved(img, "circle_outline_thick", differ=True)


@pytest.mark.parametrize("itype", (ImgType.PIL, ImgType.BGR, ImgType.RGB))
def test_draw_circle_filled_outlined_thick(itype: ImgType):
    img = Img.new(ISIZE, itype)
    img.draw.draw(CIRCLE, outline=COL, width=2, fill=COL2)
    _compared_to_saved(img, "circle_filled_outline_thick", differ=True)


@pytest.mark.parametrize("itype", (ImgType.PIL, ImgType.BGR, ImgType.RGB))
def test_draw_polygon_outlined(itype: ImgType):
    img = Img.new(ISIZE, itype)
    img.draw.polygon(POLYGON, outline=COL, width=1)
    _compared_to_saved(img, "polygon_outline", differ=False)


@pytest.mark.parametrize("itype", (ImgType.PIL, ImgType.BGR, ImgType.RGB))
def test_draw_polygon_outlined_thick(itype: ImgType):
    img = Img.new(ISIZE, itype)
    img.draw.draw(POLYGON, outline=COL, width=2)
    _compared_to_saved(img, "polygon_outline_thick", differ=True)


@pytest.mark.parametrize("itype", (ImgType.PIL, ImgType.BGR, ImgType.RGB))
def test_draw_polygon_filled_outlined_thick(itype: ImgType):
    img = Img.new(ISIZE, itype)
    img.draw.draw(POLYGON, outline=COL, width=2, fill=COL2)
    _compared_to_saved(img, "polygon_filled_outline_thick", differ=True)
