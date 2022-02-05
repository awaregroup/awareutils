from awareutils.vision.img import Img, ImgSize, ImgType
from awareutils.vision.video import ThreadedOpenCVFileVideoCapture, ThreadedOpenCVVideoWriter


def test_file_writer_and_reader(tmp_path):
    img = Img.new(size=ImgSize(h=100, w=100), itype=ImgType.BGR)
    fpath = str(tmp_path) + "tmp.avi"
    with ThreadedOpenCVVideoWriter(path=fpath, height=100, width=100, fps=30) as vo:
        for _ in range(3):
            vo.write(img)
    with ThreadedOpenCVFileVideoCapture(path=fpath) as vo:
        assert len(list(vo.read())) == 3
