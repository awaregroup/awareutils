import hashlib
from typing import Iterable, Tuple

from loguru import logger


class Col:
    named: "NamedCols"

    def __init__(self, r: int, g: int, b: int, clip: bool = False, fix_numeric_type: bool = True):
        self._clip = clip
        self._fix_numeric_type = fix_numeric_type
        self.r = r  # Note this is calling the setter
        self.g = g
        self.b = b

    @property
    def r(self) -> int:
        return self._r

    @r.setter
    def r(self, r: int) -> None:
        self._r = self._validate_uint8(r)

    @property
    def g(self) -> int:
        return self._g

    def __eq__(self, c: "Col") -> bool:
        return self.r == c.r and self.g == c.g and self.b == c.b

    @g.setter
    def g(self, g: int) -> None:
        self._g = self._validate_uint8(g)

    @property
    def b(self) -> int:
        return self._b

    @b.setter
    def b(self, b: int) -> None:
        self._b = self._validate_uint8(b)

    @property
    def rgb(self) -> Tuple[int, int, int]:
        return (self.r, self.g, self.b)

    @property
    def bgr(self) -> Tuple[int, int, int]:
        return (self.b, self.g, self.r)

    def _validate_uint8(self, c: int) -> int:
        if c is None:
            raise ValueError("Color r/g/b must not be None")

        if not isinstance(c, int):
            if self._fix_numeric_type:
                logger.debug("Color r/g/b is meant to be int, so trying to coerce to int")
                c = int(c)
            else:
                raise ValueError("Color r/g/b is meant to be int but it isn't.")

        # Should always be >= 0
        if c < 0 or c > 255:
            if self._clip:
                c = min(255, max(0, c))
                logger.debug("Color r/g/b must be 0 - 255 but it isn't, so clipping to this range.")
            else:
                raise ValueError("Color r/g/b must be 0 - 255 but it isn't.")

        # Phew, done:
        return c


def pick_col(s: str) -> Col:

    if not isinstance(s, str):
        raise RuntimeError("Please provide a string argument to pick_col")

    # Approach based on https://github.com/vaab/colour/blob/11f138eb7841d2045160b378a2eec0c2321144c0/colour.py#L737
    # i.e. hash the string representation
    digest = hashlib.md5(s.encode("utf8")).hexdigest()
    n = int(len(digest) / 3)
    mx = 2 ** (4 * n) - 1
    rgb = (int(int(digest[i * n : (i + 1) * n], 16) / mx * 256) for i in range(3))
    return Col(*rgb)


class DivergingPalette:
    def __init__(self, labels: Iterable[str] = None):
        # ColorBrewer Diverging 12-class Paired
        self._cols = (
            (166, 206, 227),
            (31, 120, 180),
            (178, 223, 138),
            (51, 160, 44),
            (251, 154, 153),
            (227, 26, 28),
            (253, 191, 111),
            (255, 127, 0),
            (202, 178, 214),
            (106, 61, 154),
            (255, 255, 153),
            (177, 89, 40),
        )
        # Create the lookup (with our own Col objects so they can be mutated)
        self._col_map = {}
        if labels is not None:
            for idx, label in enumerate(labels):
                self._col_map[label] = Col(*self._cols[idx % len(self._cols)])

    def col(self, label: str) -> Col:
        if label not in self._col_map:
            idx = len(self._col_map) % len(self._cols)
            self._col_map[label] = Col(*self._cols[idx])
        return self._col_map[label]


class NamedCols:
    alice_blue = Col(240, 248, 255)
    antique_white = Col(250, 235, 215)
    aqua = Col(0, 255, 255)
    aqua_marine = Col(127, 255, 212)
    aware_blue_dark = Col(0, 81, 155)
    aware_blue_light = Col(87, 200, 231)
    azure = Col(240, 255, 255)
    beige = Col(245, 245, 220)
    bisque = Col(255, 228, 196)
    black = Col(0, 0, 0)
    blanched_almond = Col(255, 235, 205)
    blue = Col(0, 0, 255)
    blue_violet = Col(138, 43, 226)
    brown = Col(165, 42, 42)
    burly_wood = Col(222, 184, 135)
    cadet_blue = Col(95, 158, 160)
    chart_reuse = Col(127, 255, 0)
    chocolate = Col(210, 105, 30)
    coral = Col(255, 127, 80)
    corn_flower_blue = Col(100, 149, 237)
    corn_silk = Col(255, 248, 220)
    crimson = Col(220, 20, 60)
    cyan = Col(0, 255, 255)
    dark_blue = Col(0, 0, 139)
    dark_cyan = Col(0, 139, 139)
    dark_golden_rod = Col(184, 134, 11)
    dark_gray = Col(169, 169, 169)
    dark_green = Col(0, 100, 0)
    dark_grey = Col(169, 169, 169)
    dark_khaki = Col(189, 183, 107)
    dark_magenta = Col(139, 0, 139)
    dark_olive_green = Col(85, 107, 47)
    dark_orange = Col(255, 140, 0)
    dark_orchid = Col(153, 50, 204)
    dark_red = Col(139, 0, 0)
    dark_salmon = Col(233, 150, 122)
    dark_sea_green = Col(143, 188, 143)
    dark_slate_blue = Col(72, 61, 139)
    dark_slate_gray = Col(47, 79, 79)
    dark_turquoise = Col(0, 206, 209)
    dark_violet = Col(148, 0, 211)
    deep_pink = Col(255, 20, 147)
    deep_sky_blue = Col(0, 191, 255)
    dim_gray = Col(105, 105, 105)
    dim_grey = Col(105, 105, 105)
    dodger_blue = Col(30, 144, 255)
    firebrick = Col(178, 34, 34)
    floral_white = Col(255, 250, 240)
    forest_green = Col(34, 139, 34)
    fuchsia = Col(255, 0, 255)
    gainsboro = Col(220, 220, 220)
    ghost_white = Col(248, 248, 255)
    gold = Col(255, 215, 0)
    golden_rod = Col(218, 165, 32)
    gray = Col(128, 128, 128)
    green = Col(0, 128, 0)
    green_yellow = Col(173, 255, 47)
    grey = Col(128, 128, 128)
    honeydew = Col(240, 255, 240)
    hot_pink = Col(255, 105, 180)
    indian_red = Col(205, 92, 92)
    indigo = Col(75, 0, 130)
    ivory = Col(255, 255, 240)
    khaki = Col(240, 230, 140)
    lavender = Col(230, 230, 250)
    lavender_blush = Col(255, 240, 245)
    lawn_green = Col(124, 252, 0)
    lemon_chiffon = Col(255, 250, 205)
    light_blue = Col(173, 216, 230)
    light_coral = Col(240, 128, 128)
    light_cyan = Col(224, 255, 255)
    light_golden_rod_yellow = Col(250, 250, 210)
    light_gray = Col(211, 211, 211)
    light_green = Col(144, 238, 144)
    light_grey = Col(211, 211, 211)
    light_pink = Col(255, 182, 193)
    light_salmon = Col(255, 160, 122)
    light_sea_green = Col(32, 178, 170)
    light_sky_blue = Col(135, 206, 250)
    light_slate_gray = Col(119, 136, 153)
    light_steel_blue = Col(176, 196, 222)
    light_yellow = Col(255, 255, 224)
    lime = Col(0, 255, 0)
    lime_green = Col(50, 205, 50)
    linen = Col(250, 240, 230)
    magenta = Col(255, 0, 255)
    maroon = Col(128, 0, 0)
    medium_aqua_marine = Col(102, 205, 170)
    medium_blue = Col(0, 0, 205)
    medium_orchid = Col(186, 85, 211)
    medium_purple = Col(147, 112, 219)
    medium_sea_green = Col(60, 179, 113)
    medium_slate_blue = Col(123, 104, 238)
    medium_spring_green = Col(0, 250, 154)
    medium_turquoise = Col(72, 209, 204)
    medium_violet_red = Col(199, 21, 133)
    midnight_blue = Col(25, 25, 112)
    mint_cream = Col(245, 255, 250)
    misty_rose = Col(255, 228, 225)
    moccasin = Col(255, 228, 181)
    navajo_white = Col(255, 222, 173)
    navy = Col(0, 0, 128)
    old_lace = Col(253, 245, 230)
    olive = Col(128, 128, 0)
    olive_drab = Col(107, 142, 35)
    orange = Col(255, 165, 0)
    orange_red = Col(255, 69, 0)
    orchid = Col(218, 112, 214)
    pale_golden_rod = Col(238, 232, 170)
    pale_green = Col(152, 251, 152)
    pale_turquoise = Col(175, 238, 238)
    pale_violet_red = Col(219, 112, 147)
    papaya_whip = Col(255, 239, 213)
    peach_puff = Col(255, 218, 185)
    peru = Col(205, 133, 63)
    pink = Col(255, 192, 203)
    plum = Col(221, 160, 221)
    powder_blue = Col(176, 224, 230)
    purple = Col(128, 0, 128)
    red = Col(255, 0, 0)
    rosy_brown = Col(188, 143, 143)
    royal_blue = Col(65, 105, 225)
    saddle_brown = Col(139, 69, 19)
    salmon = Col(250, 128, 114)
    sandy_brown = Col(244, 164, 96)
    sea_green = Col(46, 139, 87)
    sea_shell = Col(255, 245, 238)
    sienna = Col(160, 82, 45)
    silver = Col(192, 192, 192)
    sky_blue = Col(135, 206, 235)
    slate_blue = Col(106, 90, 205)
    slate_gray = Col(112, 128, 144)
    snow = Col(255, 250, 250)
    spring_green = Col(0, 255, 127)
    steel_blue = Col(70, 130, 180)
    tan = Col(210, 180, 140)
    teal = Col(0, 128, 128)
    thistle = Col(216, 191, 216)
    tomato = Col(255, 99, 71)
    turquoise = Col(64, 224, 208)
    violet = Col(238, 130, 238)
    wheat = Col(245, 222, 179)
    white = Col(255, 255, 255)
    white_smoke = Col(245, 245, 245)
    yellow = Col(255, 255, 0)
    yellow_green = Col(154, 205, 50)


Col.named = NamedCols
