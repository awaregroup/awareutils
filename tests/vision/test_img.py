import os
import tempfile
import time

import numpy as np
import pytest
from awareutils.vision.img import Img, ImgSize, ImgType
from awareutils.vision.shape import Rectangle
from PIL import Image as PILImage

xfail = pytest.mark.xfail

EMPTY_ARRAY = np.zeros((10, 10, 3), np.uint8)
EMPTY_PIL = PILImage.new("RGB", (10, 10))


class TempFilePath(object):
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __enter__(self):
        self.handle, self.path = tempfile.mkstemp(*self.args, **self.kwargs)
        os.close(self.handle)
        return self

    def __exit__(self, *args, **kwargs):
        if os.path.exists(self.path):
            os.remove(self.path)
        time.sleep(0.1)


def test_isize_constructor():
    ImgSize(w=100, h=100)
    with pytest.raises(Exception):
        ImgSize(100, 100)


def test_rgb_constructor():
    Img(source=EMPTY_ARRAY, itype=ImgType.RGB)
    Img(EMPTY_ARRAY, ImgType.RGB)


def test_bgr_constructor():
    Img(source=EMPTY_ARRAY, itype=ImgType.BGR)
    Img(EMPTY_ARRAY, ImgType.BGR)


def test_pil_constructor():
    Img(source=EMPTY_PIL, itype=ImgType.PIL)
    Img(EMPTY_PIL, ImgType.PIL)


def test_from_bgr():
    img = Img.from_bgr(EMPTY_ARRAY)
    assert img.itype == ImgType.BGR


def test_from_rgb():
    img = Img.from_rgb(EMPTY_ARRAY)
    assert img.itype == ImgType.RGB


@pytest.mark.parametrize("fmt", ["JPEG", "PNG"])
@pytest.mark.parametrize("itype_save", [ImgType.BGR, ImgType.RGB, ImgType.PIL])
@pytest.mark.parametrize("itype_open", [ImgType.BGR, ImgType.RGB, ImgType.PIL])
@pytest.mark.parametrize("do_metadata", [True, False])
def test_save_load_metadata(fmt, itype_save, itype_open, do_metadata):

    metadata = [1, 2, 3] if do_metadata else None
    if itype_save == ImgType.BGR:
        img = Img.from_bgr(EMPTY_ARRAY, metadata=metadata)
    elif itype_save == ImgType.RGB:
        img = Img.from_rgb(EMPTY_ARRAY, metadata=metadata)
    elif itype_save == ImgType.PIL:
        img = Img.from_pil(EMPTY_PIL, metadata=metadata)
    with tempfile.NamedTemporaryFile() as f:
        f.close()  # 'cos it opens it opened
        fpath = f"{f.name}.{fmt.lower()}"
        img.save(fpath, format=fmt, save_metadata=do_metadata)
        img = Img.open(fpath, itype=itype_open, load_metadata=do_metadata)
        assert img.itype == itype_open
        if do_metadata:
            assert img.metadata == metadata
        else:
            assert img.metadata is None


@pytest.mark.parametrize("itype", [ImgType.BGR, ImgType.RGB, ImgType.PIL])
@pytest.mark.parametrize("meta", [True, False])
def test_img_resize(itype: ImgType, meta: bool):
    # Check resizes, preserves metadata etc.
    s0 = ImgSize(h=100, w=100)
    s1 = ImgSize(h=10, w=10)
    img = Img.new(size=s0, itype=itype)
    if meta:
        img.metadata = {"a": 1}
    resized = img.resize(s1)
    assert resized.h == s1.h
    assert resized.w == s1.w
    if meta:
        assert resized.metadata == {"a": 1}


@pytest.mark.parametrize("itype", [ImgType.BGR, ImgType.RGB, ImgType.PIL])
@pytest.mark.parametrize("meta", [True, False])
@pytest.mark.parametrize("copy", [True, False])
@pytest.mark.parametrize("wrong_crop_size", [True, False])
def test_img_crop(itype: ImgType, meta: bool, copy: bool, wrong_crop_size: bool):
    size = ImgSize(h=100, w=100)
    img = Img.new(size=size, itype=itype)
    if meta:
        img.metadata = {"a": 1}
    rect = Rectangle(x0=10, y0=10, x1=19, y1=19, isize=ImgSize(h=99, w=99) if wrong_crop_size else size)
    if itype == ImgType.PIL and not copy:
        # Can never ask not to copy for PIL
        with pytest.raises(Exception):
            cropped = img.crop(rect, copy=copy)
    else:
        if wrong_crop_size:
            with pytest.raises(Exception):
                cropped = img.crop(rect, copy=copy)
        else:
            cropped = img.crop(rect, copy=copy)
            assert cropped.h == 10 and cropped.w == 10
            if meta:
                assert cropped.metadata == {"a": 1}


@pytest.mark.parametrize("itype", [ImgType.BGR, ImgType.RGB, ImgType.PIL])
@pytest.mark.parametrize("meta", [True, False])
def test_img_new(itype: ImgType, meta: bool):
    if meta:
        img = Img.new(size=ImgSize(h=10, w=10), itype=itype)
    else:
        img = Img.new(size=ImgSize(h=10, w=10), itype=itype, metadata={"a": 1})
    assert img is not None
