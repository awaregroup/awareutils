import collections
import platform
import threading
import time
from abc import ABCMeta, abstractmethod, abstractproperty
from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from typing import Any, Iterator, Union

from awareutils.vision._threading import Threadable
from awareutils.vision.img import Img
from loguru import logger

# Import only what we need
try:
    import cv2
except ImportError:
    from awareutils.vision.mock import cv2


@dataclass
class CameraFrame:
    fidx: int
    img: Img


class NoMoreFrames(Exception):
    pass


@dataclass
class FPS:
    last_frame_duration_ms: float
    smoothed_duration_ms: float

    @property
    def last_frame_fps(self) -> float:
        if self.last_frame_duration_ms is None:
            return None
        elif self.last_frame_duration_ms == 0:
            return float("Inf")
        else:
            return 1 / self.last_frame_duration_ms * 1000

    @property
    def smoothed_fps(self) -> float:
        if self.smoothed_duration_ms is None:
            return None
        elif self.smoothed_duration_ms == 0:
            return float("Inf")
        else:
            return 1 / self.smoothed_duration_ms * 1000


def _calculate_fps(times) -> FPS:
    last_duration_ms = None
    smoothed_duration_ms = None
    if len(times) >= 2:
        last_duration_ms = (times[-1] - times[-2]) * 1000
        smoothed_duration_ms = (times[-1] - times[0]) / (len(times) - 1) * 1000
    return FPS(last_frame_duration_ms=last_duration_ms, smoothed_duration_ms=smoothed_duration_ms)


class VideoCapture(metaclass=ABCMeta):
    """
    Class that reads a generic camera in a separate thread.
    """

    @abstractmethod
    def open(self):
        pass

    @abstractmethod
    def close(self, *args, **kwargs):
        pass

    @abstractmethod
    def read(self, *args, **kwargs):
        pass

    @property
    @abstractproperty
    def width(self):
        pass

    @property
    @abstractproperty
    def height(self):
        pass

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args, **kwargs):
        self.close()


class ThreadedVideoCapture(Threadable, VideoCapture, metaclass=ABCMeta):
    def __init__(self, non_skipping: bool, finite: bool, simulated_read_fps: float = None, *args, **kwargs):
        # TODO: check types or args
        super().__init__(*args, **kwargs)
        self._non_skipping = non_skipping
        self._finite = finite
        self._simulated_read_fps = simulated_read_fps
        self._current_img = None
        self._fidx = None
        self._running = False
        self._capture_open_event = threading.Event()
        self._first_frame_event = threading.Event()
        self._new_frame_or_no_more_frames_event = threading.Event()
        self._frame_yielded_event = threading.Event()
        self._height = None
        self._width = None
        self._no_more_frames = False
        self._read_times = collections.deque([], maxlen=10)
        self._yield_times = collections.deque([], maxlen=10)

    @abstractmethod
    def _open_capture(self, *args, **kwargs) -> bool:
        """
        Open the camera, set up any properties, etc.
        """
        pass

    @abstractmethod
    def _read_frame(self, *args, **kwargs) -> Img:
        """
        To read the next frame.
        """
        pass

    @abstractmethod
    def _close_capture(self, *args, **kwargs) -> bool:
        pass

    @abstractmethod
    def _get_height(self) -> int:
        pass

    @abstractmethod
    def _get_width(self) -> int:
        pass

    def _close_immediately(self) -> None:
        self._running = False

    def _close_after_thread_finished(self) -> None:
        self._read_times.clear()
        self._yield_times.clear()
        self._fidx = None
        self._current_img = None

    def _run(self):
        # With OpenCV, everything has to happen in one thread, so we need to open, read, and close, all in this loop.
        self._running = True
        logger.info("Trying to open camera")
        try:
            self._open_capture()
            self._capture_open_event.set()
        except:  # NOQA
            logger.exception("Failed to open camera!")
            self._stop()
            self._running = False

        self._frame_yielded_event.set()
        last_new_frame_triggered = None
        self._fidx = -1
        while self._running:
            try:
                img = self._read_frame()
                self._read_times.append(time.time())
                # If we're non-skipping, we don't want to read until we've yielded the last one - usually when we're
                # reading a video we don't want to skip frames. Note that we do this after reading (which can be
                # expensive computationally)
                if self._non_skipping:
                    self._block_until(self._frame_yielded_event)
                    self._frame_yielded_event.clear()
                self._fidx += 1
                self._current_img = img
                self._first_frame_event.set()
                # OK, if the user wants to simulate a specific read FPS, sleep here before we notify that there's a new
                # frame:
                t = time.time()
                if self._simulated_read_fps and last_new_frame_triggered is not None:
                    sleep_duration = 1 / self._simulated_read_fps - t
                    if sleep_duration < 0:
                        logger.warning(
                            "You wanted to simulate an FPS of {fps} but reading from the camera is slower than this.",
                            fps=self._simulated_read_fps,
                        )
                    else:
                        time.sleep(sleep_duration)
                # Cool, tell the world there's a new frame:
                self._new_frame_or_no_more_frames_event.set()
                last_new_frame_triggered = t
            except NoMoreFrames:
                # Don't alert about this if we're a finite capture (e.g. a video file), which should finish ...
                if not self._finite:
                    logger.exception("Failed to read frame!")
                # Trigger one last new_frame_or_no_more_frames_event so our read() function can unblock ... note that we
                # should really tidy this up a bit with condition etc. and don't give our events two purposes
                self._no_more_frames = True
                self._new_frame_or_no_more_frames_event.set()
                break
            except:  # NOQA
                logger.exception("General read failure!")
                break

        # Close here in thread
        self._running = False
        self._close_capture()

    @property
    def height(self):
        self._block_until(self._capture_open_event, timeout=5)
        return self._get_height()

    @property
    def width(self):
        self._block_until(self._capture_open_event, timeout=5)
        return self._get_width()

    def read(self, timeout: int = 5) -> Iterator[CameraFrame]:
        self._block_until(self._first_frame_event, timeout=timeout)
        last_fidx = None
        while True:
            # Block until we get a new frame (so we don't read the same frame twice)
            self._block_until(self._new_frame_or_no_more_frames_event, timeout=timeout)

            # If we're finished, good:
            if self._no_more_frames:
                return

            self._new_frame_or_no_more_frames_event.clear()
            fidx, img = self._fidx, self._current_img
            # check for frame skip:
            if last_fidx is not None:
                if last_fidx == fidx:
                    raise RuntimeError("Read the same frame twice - this shouldn't happen!")
                elif (fidx - last_fidx) > 1:
                    logger.debug(
                        (
                            "Frame skip! Last frame yielded was {last} and we're on {fidx} i.e. {n} were skipped. This "
                            "indicates your processing is slower than the camera read FPS, so consider slowing down "
                            "the camera framerate to avoid skips, or speed up your processing."
                        ),
                        last=last_fidx,
                        fidx=fidx,
                        n=fidx - last_fidx - 1,
                    )
            last_fidx = fidx
            self._frame_yielded_event.set()
            self._yield_times.append(time.time())
            yield CameraFrame(fidx=fidx, img=img)

    def read_fps(self) -> FPS:
        return _calculate_fps(self._read_times)

    def yielded_fps(self) -> FPS:
        return _calculate_fps(self._yield_times)


class ThreadedOpenCVLiveVideoCapture(ThreadedVideoCapture):
    def __init__(self, device: Any, height: int = None, width: int = None, fps: int = None, api: int = None):
        super().__init__(finite=False, non_skipping=False, simulated_read_fps=False)
        # TODO: check they're the right types
        self._device = device
        self._intended_height = height
        self._intended_width = width
        self._intended_fps = fps
        self._api = api
        self._vi = None
        self._width = None
        self._height = None

    def _open_capture(self) -> bool:
        if platform.system().lower() == "windows" and self._api is None:
            logger.warning(
                (
                    "You're on Windows trying to read a USB device so opening with cv2.CAP_DSHOW. You might want to try"
                    " api=cv2.CAP_MSMF if that doesn't work well for you."
                )
            )
            vi = cv2.VideoCapture(self._device, cv2.CAP_DSHOW)
        else:
            vi = cv2.VideoCapture(self._device)

        if not vi.isOpened():
            raise RuntimeError("Failed to open OpenCV camera")

        # OK, try to configure USB sources if required. Note that we set all the things first as in some cases if you
        # set e.g. the width alone, it won't actually update until the width and height have been set.
        for name, prop, intended in (
            ("height", cv2.CAP_PROP_FRAME_HEIGHT, self._intended_height),
            ("width", cv2.CAP_PROP_FRAME_WIDTH, self._intended_width),
            ("fps", cv2.CAP_PROP_FPS, self._intended_fps),
        ):
            if intended is not None:
                logger.info("Trying to set camera {name} to {intended}", name=name, intended=intended)
                vi.set(prop, intended)

        # Now check the properties have been set:
        for name, prop, intended in (
            ("height", cv2.CAP_PROP_FRAME_HEIGHT, self._intended_height),
            ("width", cv2.CAP_PROP_FRAME_WIDTH, self._intended_width),
            ("fps", cv2.CAP_PROP_FPS, self._intended_fps),
        ):
            if intended is not None:
                v = int(vi.get(prop))
                logger.info(
                    "Camera {name} is {actual} (intended was {intended})", name=name, actual=v, intended=intended
                )
                if v != intended:
                    raise RuntimeError(f"Couldn't set camera {name} to {intended}")

        # Do this in here - opencv won't let us access this outside the thread, so we can't do `self._vi.get(...)` in
        # the main loop, and hence we set it here for our main props to read.
        self._width = int(vi.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(vi.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self._vi = vi

    def _get_height(self) -> int:
        assert self._height is not None, "height prop should block until first frame, where self._height is set"
        return self._height

    def _get_width(self) -> int:
        assert self._width is not None, "width prop should block until first frame, where self._width is set"
        return self._width

    def _read_frame(self):
        ok, bgr = self._vi.read()
        if not ok:
            raise RuntimeError("Couldn't get any more frames!")
        return Img.from_bgr(bgr)

    def _close_capture(self):
        if self._vi is not None:
            self._vi.release()


class ThreadedOpenCVFileVideoCapture(ThreadedVideoCapture):
    def __init__(self, path: Union[Path, str], simulated_read_fps: int = None):
        # TODO: check types or args
        if simulated_read_fps is None:
            super().__init__(finite=True, non_skipping=True, simulated_read_fps=None)
        else:
            super().__init__(finite=True, non_skipping=False, simulated_read_fps=None)
        if not isinstance(path, (Path, str)):
            raise ValueError("path must be a Path or str")
        self._path = str(path)
        self._vi = None

    def _open_capture(self) -> bool:
        self._vi = cv2.VideoCapture(self._path)
        # See comment above why we're doing this here:
        self._width = int(self._vi.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self._vi.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def _get_height(self) -> int:
        assert self._height is not None, "height prop should block until first frame, where self._height is set"
        return self._height

    def _get_width(self) -> int:
        assert self._width is not None, "width prop should block until first frame, where self._width is set"
        return self._width

    def _read_frame(self):
        ok, bgr = self._vi.read()
        if not ok:
            raise NoMoreFrames()
        return Img.from_bgr(bgr)

    def _close_capture(self):
        if self._vi is not None:
            self._vi.release()


class VideoWriter(metaclass=ABCMeta):
    """
    Generic class for writing video.
    """

    @abstractmethod
    def open(self):
        pass

    @abstractmethod
    def close(self, *args, **kwargs):
        pass

    @abstractmethod
    def write(self, img: Img):
        pass

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args, **kwargs):
        self.close()


class ThreadedVideoWriter(Threadable, VideoWriter, metaclass=ABCMeta):
    """
    Class that writes video in a separate thread. The main change here is that we ensure we don't close while we're in
    the middle of writing.

    TODO: handle the case where we can't write fast enough and the queue keeps growing. We could also have a version
    where we only write the latest frame (and who cares if we skip a few).
    """

    def __init__(self):
        super().__init__()
        self._q = Queue()
        self._write_times = collections.deque([], maxlen=10)
        self._running = False

    @abstractmethod
    def _open_in_thread(self, *args, **kwargs):
        pass

    def write(self, img: Img):
        if not self._running:
            raise RuntimeError("ThreadedVideoWriter is not running, so can't write to it. Check the logs.")
        self._q.put((False, img))  # First arg is whether we want to stop or not.

    @abstractmethod
    def _write_in_thread(self, img: Img):
        pass

    def _run(self):
        self._running = True
        try:
            self._open_in_thread()
            while self._running:  # Note - this isn't interruptible as we clear with a task
                stop, img = self._q.get()
                # TODO: errors if q is getting too long
                if stop:
                    break
                else:
                    self._write_in_thread(img)
                    self._write_times.append(time.time())
        except:  # NOQA
            logger.exception("Failed while writing!")
            self._running = False
        self._close_in_thread()

    @abstractmethod
    def _close_in_thread(self, *args, **kwargs):
        pass

    def _close_immediately(self):
        # TODO: do we empty the queue?
        # Add a 'stop' item to the queue - this is our way of telling the _run function to stop processing. (It's a bit
        # of a hack for now to stop the _run method hanging on self._q.get(). TODO make this nicer = ) ) Note that this
        # has the side effect of clearing out the queue before we close (as we're putting on the end)
        self._q.put((True, None))
        # Note that our parent Threadable will block until this main thread has finished

    # TODO: add the requested write FPS?

    def _close_after_thread_finished(self) -> None:
        pass

    def write_fps(self) -> FPS:
        return _calculate_fps(self._write_times)


class ThreadedOpenCVVideoWriter(ThreadedVideoWriter):
    def __init__(self, path: Union[Path, str], height: int, width: int, fps: int):
        super().__init__()
        if not isinstance(path, (Path, str)):
            raise ValueError("path must be a Path or str")
        self.path = str(path)
        self.height = height
        self.width = width
        self.fps = fps
        self._vo = None

    def _open_in_thread(self, *args, **kwargs):
        self._vo = cv2.VideoWriter(self.path, cv2.VideoWriter_fourcc(*"XVID"), self.fps, (self.width, self.height))
        if not self._vo.isOpened():
            raise RuntimeError("Failed to open writer!")

    def _write_in_thread(self, img: Img):
        self._vo.write(img.bgr())

    def _close_in_thread(self):
        self._vo.release()
