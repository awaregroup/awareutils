import numpy as np
import pytest
from awareutils.vision.img_size import ImgSize
from awareutils.vision.shape import Pixel, Polygon, Rectangle

IMG_SIZE = ImgSize(h=100, w=100)
IMG_SIZE2 = ImgSize(h=1000, w=1000)

xfail = pytest.mark.xfail

# Pixel


def test_pixel_constructor():
    Pixel(x=0, y=0, img_size=IMG_SIZE)
    # Fails without keywords
    with pytest.raises(Exception):
        Pixel(0, 0, IMG_SIZE)


def test_pixel_oob_coordinates():
    with pytest.raises(Exception):
        Pixel(x=IMG_SIZE.x, y=IMG_SIZE.y, img_size=IMG_SIZE)


def test_pixel_h():
    p = Pixel(x=0, y=0, img_size=IMG_SIZE)
    assert p.h == 1


def test_pixel_w():
    p = Pixel(x=0, y=0, img_size=IMG_SIZE)
    assert p.w == 1


def test_pixel_center():
    p = Pixel(x=0, y=0, img_size=IMG_SIZE)
    assert p.center.x == p.x and p.center.y == p.y


def test_pixel_area():
    p = Pixel(x=0, y=0, img_size=IMG_SIZE)
    assert p.area == 1


def test_pixel_project():
    p = Pixel(x=0, y=0, img_size=IMG_SIZE)
    p2 = p.project(img_size=IMG_SIZE2)
    assert p2.x == 0 and p2.y == 0
    p = Pixel(x=1, y=1, img_size=IMG_SIZE)
    p2 = p.project(img_size=IMG_SIZE2)
    assert p2.x == 10 and p2.y == 10


# Rectangle


def test_rectangle_constructor():
    Rectangle(x0=0, y0=0, x1=10, y1=10, img_size=IMG_SIZE)
    # Fails without keywords
    with pytest.raises(Exception):
        Rectangle(0, 0, 10, 10, IMG_SIZE)


def test_rectangle_oob_coordinates():
    with pytest.raises(Exception):
        Rectangle(x0=0, y0=0, x1=IMG_SIZE.x, y1=IMG_SIZE.y, img_size=IMG_SIZE)


def test_rectangle_h():
    r = Rectangle(x0=0, y0=0, x1=10, y1=10, img_size=IMG_SIZE)
    assert r.h == 11


def test_rectangle_w():
    r = Rectangle(x0=0, y0=0, x1=10, y1=10, img_size=IMG_SIZE)
    assert r.w == 11


def test_rectangle_center():
    r = Rectangle(x0=0, y0=0, x1=10, y1=10, img_size=IMG_SIZE)
    assert r.center.x == 5 and r.center.y == 5


def test_rectangle_area():
    r = Rectangle(x0=0, y0=0, x1=10, y1=10, img_size=IMG_SIZE)
    assert r.area == 11 * 11


def test_rectangle_slice():
    arr = np.zeros((IMG_SIZE.h, IMG_SIZE.w, 3), np.uint8)
    r = Rectangle(x0=0, y0=0, x1=10, y1=10, img_size=IMG_SIZE)
    r.slice_array(arr)[...] = 1
    assert arr.sum() == 3 * 11 * 11


def test_rectangle_slice_wrong_size():
    arr = np.zeros((10, 10, 3), np.uint8)
    r = Rectangle(x0=0, y0=0, x1=10, y1=10, img_size=IMG_SIZE)
    with pytest.raises(Exception):
        r.slice_array(arr)


def test_rectangle_project():
    r = Rectangle(x0=0, y0=0, x1=10, y1=10, img_size=IMG_SIZE)
    p2 = r.project(img_size=IMG_SIZE2)
    assert p2.x0 == 0 and p2.y0 == 0 and p2.x1 == 100 and p2.y1 == 100


@xfail
def test_rectangle_intersection():
    raise NotImplementedError()


@xfail
def test_rectangle_no_intersection():
    raise NotImplementedError()


@xfail
def test_rectangle_iou():
    raise NotImplementedError()


@xfail
def test_rectangle_iou_no_intersection():
    raise NotImplementedError()


# Polygon


@xfail
def test_polygon():
    # Some polygon tests ...
    raise NotImplementedError()


@xfail
def test_fill():
    # Check fill fails correctly with different image sizes
    raise NotImplementedError()
