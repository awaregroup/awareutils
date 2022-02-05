from awareutils.vision.img import Img, ImgSize, ImgType
from awareutils.vision.video import ThreadedOpenCVFileVideoCapture, ThreadedOpenCVVideoWriter


def test_file_writer_and_reader(tmp_path):
    img = Img.new(size=ImgSize(h=1080, w=1920), itype=ImgType.BGR)
    fpath = str(tmp_path) + "tmp.avi"
    with ThreadedOpenCVVideoWriter(path=fpath, height=1080, width=1920, fps=30) as vo:
        for _ in range(3):
            vo.write(img)
    with ThreadedOpenCVFileVideoCapture(path=fpath) as vi:
        frames = []
        for frame in vi.read():
            frames.append(frame)
            if len(frames) == 2:
                assert isinstance(vi.read_fps().last_frame_fps, float)
                assert isinstance(vi.read_fps().smoothed_fps, float)
                assert isinstance(vi.yielded_fps().last_frame_fps, float)
                assert isinstance(vi.yielded_fps().smoothed_fps, float)
    assert len(frames) == 3
