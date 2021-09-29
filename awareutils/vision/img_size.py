class ImgSize:
    def __init__(self, *, h: int, w: int):
        self._validate_img_shape(w=w, h=h)
        self.w = w
        self.h = h

    def __eq__(self, other):
        return self.w == other.w and self.h == other.h

    @staticmethod
    def _validate_img_shape(*, w: int, h: int):
        if not isinstance(w, int) or not isinstance(h, int):
            raise ValueError("w and h should be ints")
        if w <= 1 or h <= 1:
            raise ValueError("img should be at least 1x1")
