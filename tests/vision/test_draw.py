from pathlib import Path

import numpy as np
import pytest
from awareutils.vision.col import Col
from awareutils.vision.img import Img, ImgSize, ImgType
from awareutils.vision.shape import Pixel, Rectangle

DRAW_DIR = Path(__file__).resolve().parent / "draws"
ISIZE = ImgSize(h=100, w=100)
PIXEL = Pixel(x=10, y=10, isize=ISIZE)
RECT = Rectangle(p0=PIXEL, p1=Pixel(x=20, y=20, isize=ISIZE))

xfail = pytest.mark.xfail


def test_draw_pixel_pil():
    img = Img.new_pil(ISIZE)
    pixel = Pixel(x=10, y=10, isize=ISIZE)
    img.draw.draw(pixel, fill=Col(1, 1, 1))
    assert img.bgr().sum() == 3


@pytest.mark.parametrize("itype", (ImgType.PIL, ImgType.BGR, ImgType.RGB))
def test_draw_pixel(itype: ImgType):
    img = Img.new(ISIZE, itype)
    img.draw.draw(PIXEL, fill=Col(1, 2, 3))
    rgb = img.rgb()
    assert np.array_equal(rgb[10, 10, :], (1, 2, 3))


def _compared_to_saved(img: Img, name: str, save: bool = False):
    drawtype = "pil" if img.itype == ImgType.PIL else "cv"
    expected = DRAW_DIR / f"{name}_{drawtype}.png"
    if save:
        img.save(str(expected)[:-4] + ".save.png")
    expected = Img.open_rgb(expected)
    assert np.array_equal(img.rgb(), expected.rgb())


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
    img.draw.draw(RECT, outline=Col(255, 255, 255), width=1)
    rgb = img.rgb()
    assert rgb.sum() == (11 * 2 + 9 * 2) * 3 * 255


@pytest.mark.parametrize("itype", (ImgType.PIL, ImgType.BGR, ImgType.RGB))
def test_draw_rectangle_outlined_thick(itype: ImgType):
    img = Img.new(ISIZE, itype)
    img.draw.draw(RECT, outline=Col(255, 255, 255), width=2)
    _compared_to_saved(img, "rectangle_outline_thick")


@pytest.mark.parametrize("itype", (ImgType.PIL, ImgType.BGR, ImgType.RGB))
def test_draw_rectangle_filled_outlined_thick(itype: ImgType):
    img = Img.new(ISIZE, itype)
    img.draw.draw(RECT, outline=Col.named.white, width=2, fill=Col.named.red)
    _compared_to_saved(img, "rectangle_filled_outline_thick")


@xfail
def test_all_the_shapes():
    # fill, outline, etc.
    raise NotImplementedError()
