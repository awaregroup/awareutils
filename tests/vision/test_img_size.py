import pytest
from awareutils.vision.img_size import ImgSize


def test_constructor():
    ImgSize(w=100, h=100)
    # Fails without keywords
    with pytest.raises(Exception):
        ImgSize(100, 100)
