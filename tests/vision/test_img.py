import os
import tempfile
import time
from json import load

import numpy as np
import pytest
from awareutils.vision.img import Img, ImgType
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
@pytest.mark.parametrize("load_metadata", [True, False])
def test_save_load_metadata(fmt, itype_save, itype_open, load_metadata):

    metadata = [1, 2, 3]
    if itype_save == ImgType.BGR:
        img = Img.from_bgr(EMPTY_ARRAY, metadata=metadata)
    elif itype_save == ImgType.RGB:
        img = Img.from_rgb(EMPTY_ARRAY, metadata=metadata)
    elif itype_save == ImgType.PIL:
        img = Img.from_pil(EMPTY_PIL, metadata=metadata)
    # fpath = os.path.join(tempfile.gettempdir(), os.urandom(24).hex())
    # with TempFilePath(suffix="." + fmt) as f:
    # with tempfile.TemporaryDirectory() as dirname:
    with tempfile.NamedTemporaryFile() as f:
        f.close()  # 'cos it opens it opened
        fpath = f.name
        img.save(fpath, format=fmt)
        img = Img.open(fpath, itype=itype_open, load_metadata=load_metadata)
        assert img.itype == itype_open
        if load_metadata:
            assert img.metadata == metadata
        else:
            assert img.metadata is None


@xfail
def test_img_resize():
    # Check resizes, preserves metadata etc.
    raise NotImplementedError()


@xfail
def test_img_crop():
    # Check resizes, preserves metadata etc.
    raise NotImplementedError()
