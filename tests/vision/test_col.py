from awareutils.vision.col import Col, DivergingPalette, pick_col


def test_white():
    white = Col.named.white
    assert white.r == 255
    assert white.g == 255
    assert white.b == 255


def test_col_is_rgb_args():
    col = Col(1, 2, 3)
    assert col.r == 1
    assert col.g == 2
    assert col.b == 3


def test_diverging_pallete_specified_labels():
    p = DivergingPalette(labels=[str(i) for i in range(100)])
    assert p.col("0") != p.col("1")
    assert p.col("0") == p.col("12")


def test_diverging_pallete_unspecified_labels():
    p = DivergingPalette()
    for i in range(100):
        p.col(str(i))
    assert p.col("0") != p.col("1")
    assert p.col("0") == p.col("12")


def test_pick_col():
    cols = [pick_col(str(i)) for i in range(1000)]
    for c in ("r", "g", "b"):
        col = [getattr(col, c) for col in cols]
        assert min(col) == 0
        assert max(col) == 255
