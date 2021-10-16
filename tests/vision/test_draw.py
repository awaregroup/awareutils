import pytest
from awareutils.vision.col import Col
from awareutils.vision.img import Img, ImgSize, ImgType
from awareutils.vision.shape import Pixel

isize = ImgSize(h=100, w=100)

xfail = pytest.mark.xfail


def test_draw_pixel_pil():
    img = Img.new_pil(isize)
    pixel = Pixel(x=10, y=10, isize=isize)
    img.draw.draw(pixel, fill=Col(1, 1, 1))
    assert img.bgr().sum() == 3


def test_rgb_vs_bgr():
    for itype in (ImgType.PIL, ImgType.BGR, ImgType.RGB):
        img = Img.new(isize, itype)
        pixel = Pixel(x=10, y=10, isize=isize)
        img.draw.draw(pixel, fill=Col(1, 2, 3))
        assert img.rgb()[10, 10, 0] == 1
        assert img.rgb()[10, 10, 1] == 2
        assert img.rgb()[10, 10, 2] == 3


@xfail
def test_draw_pixel_rgb():
    raise NotImplementedError()


@xfail
def test_all_the_shapes():
    # fill, outline, etc.
    raise NotImplementedError()
