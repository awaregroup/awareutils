import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from queue import Queue
from tkinter import W
from typing import Callable, List, Tuple, Union

import cv2
import numpy as np
from awareutils.vision.col import Col
from awareutils.vision.img import Img, ImgSize
from awareutils.vision.shape import Pixel, Rectangle
from loguru import logger


@dataclass
class ConsoleText:
    text: str
    font_height: float = 0.015
    col: Col = Col.named.white


@dataclass
class _DrawTask:
    img: Img
    stop: bool = False
    console_texts: List[ConsoleText] = field(default_factory=[])
    delay_ms: float = 1


def default_keyboard_callback(key: int) -> bool:
    # Return whether or not to finish
    if chr(key) == "q":
        logger.info("'q' pressed - closing")
        return True
    return False


class OpenCVGUI:
    def __init__(
        self,
        window_name: str = None,
        keyboard_callback: Callable = None,
        mouse_callback: Callable = None,
        min_console_ppn: float = 0,
        console_font: int = cv2.FONT_HERSHEY_PLAIN,
        console_col: Col = Col.named.black,
        padding_col: Col = Col.named.black,
    ):
        """
        If min_console_ppn is > 0 it means we want to have an area beside (not over top of) the image for putting
        random text information. If the image is too wide for the screen, then the console will be at the bottom
        (instead of padding top and bottom) ... with a minimum size of min_console_ppn of the height. Likewise if the
        image is too high, we pad on the sides, etc.
        """
        self._window_name = window_name
        self._mouse_callback = mouse_callback
        self._keyboard_callback = default_keyboard_callback if keyboard_callback is None else keyboard_callback
        self._window_setup = False
        self._min_console_ppn = min_console_ppn
        self._console_col = console_col
        self._console_font = console_font
        self._padding_col = padding_col
        self._out: Img = None
        self._out_img_rect: Rectangle = None
        self._out_console_rect: Rectangle = None
        self._q = Queue()
        self._running = False
        self._thread = None
        self._closed = threading.Event()

    def _setup_window(self, img_size: ImgSize):
        # TODO: allow non fullscreen
        # The "right" way to do fullscreen windows:
        cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(self._window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.moveWindow(self._window_name, 0, 0)
        # Register our callback
        if self._mouse_callback is not None:
            cv2.setMouseCallback(self._window_name, self._mouse_callback)

        # Get aspect ratio and calculate padding we need, inc. console area
        _, _, window_w, window_h = cv2.getWindowImageRect(self._window_name)
        window_size = ImgSize(w=window_w, h=window_h)
        logger.debug("Window size: {window_size}", window_size=window_size)
        logger.debug("Image size: {img_size}", img_size=img_size)
        window_aspect_ratio = window_size.w / window_size.h
        img_aspect_ratio = img_size.w / img_size.h

        # # If we're adding a console, we just pretend the image is bigger by that much
        # if self._min_console_ppn > 0:
        #     img_size = ImgSize(w=int(img_size.w * (1 + self._min_console_ppn)), h=img_size.h)
        #     img_aspect_ratio = img_size.w / img_size.h
        # Now do the padding.
        if img_aspect_ratio >= window_aspect_ratio:
            # Img is wider than the screen, so pad top and bottom
            drawn_h = int(window_size.h / img_aspect_ratio)
            if self._min_console_ppn > 0:
                # If we're doing a console, draw the image at the top, and claim everything underneath as console.
                pad_ppn = (window_size.h - drawn_h) / window_size.h
                img_x0 = 0
                img_x1 = window_size.w - 1
                if pad_ppn < self._min_console_ppn:
                    # OK, got to bump it so we get more space. This also means we need to reduce the size of the image
                    # in width too ...
                    new_drawn_h = window_size.h * (1 - self._min_console_ppn)
                    new_drawn_w = new_drawn_h / drawn_h * window_size.w
                    img_x0 = int((window_size.w - new_drawn_w) / 2)
                    img_x1 = img_x0 + new_drawn_w - 1
                    drawn_h = new_drawn_h
                drawn_img_rect = Rectangle(
                    p0=Pixel(x=img_x0, y=0, isize=window_size),
                    p1=Pixel(x=img_x1, y=drawn_h - 1, isize=window_size),
                )
                console_rect = Rectangle(
                    p0=Pixel(x=0, y=drawn_h, isize=window_size),
                    p1=Pixel(x=window_size.w - 1, y=window_size.h - 1, isize=window_size),
                )
            else:
                pad_top = int((window_size.h - drawn_h) / 2)
                drawn_img_rect = Rectangle(
                    p0=Pixel(x=0, y=pad_top, isize=window_size),
                    p1=Pixel(x=window_size.w - 1, y=pad_top + drawn_h - 1, isize=window_size),
                )
                console_rect = None
        else:
            # Img is narrower than screen, so pad sides.
            drawn_w = int(window_size.h * img_aspect_ratio)
            if self._min_console_ppn > 0:
                # If we're adding a console, put the image on the left and claim everything to the right as console
                pad_ppn = (window_size.w - drawn_w) / window_size.w
                img_y0 = 0
                img_y1 = window_size.h - 1
                if pad_ppn < self._min_console_ppn:
                    # OK, got to bump it so we get more space. This also means we need to reduce the size of the image
                    # in height too ...
                    new_drawn_w = window_size.w * (1 - self._min_console_ppn)
                    new_drawn_h = new_drawn_w / drawn_w * window_size.h
                    img_y0 = int((window_size.h - new_drawn_h) / 2)
                    img_y1 = img_y0 + new_drawn_h - 1
                    drawn_w = new_drawn_w
                drawn_img_rect = Rectangle(
                    p0=Pixel(x=0, y=img_y0, isize=window_size),
                    p1=Pixel(x=drawn_w - 1, y=img_y1, isize=window_size),
                )
                console_rect = Rectangle(
                    p0=Pixel(x=drawn_w, y=0, isize=window_size),
                    p1=Pixel(x=window_size.w - 1, y=window_size.h - 1, isize=window_size),
                )
            else:
                pad_left = int((window_size.w - drawn_w) / 2)
                drawn_img_rect = Rectangle(
                    p0=Pixel(x=pad_left, y=0, isize=window_size),
                    p1=Pixel(x=pad_left + drawn_w - 1, y=window_size.h - 1, isize=window_size),
                )
                console_rect = None

        logger.debug("drawn_img_rect: {drawn_img_rect}", drawn_img_rect=drawn_img_rect)
        logger.debug("console_rect: {console_rect}", console_rect=console_rect)

        # Save things
        self._out = Img.new_bgr(window_size, col=self._padding_col)
        self._out_img_rect = drawn_img_rect
        self._out_console_rect = console_rect
        self._window_setup = True

    def _threaded_draw(self, task: _DrawTask) -> bool:
        """
        Return true if we should quit the drawing loop (based on keyboard interaction)
        """
        assert not task.stop

        # Clear out console:
        self._out_console_rect.slice_array(self._out.bgr())[:, :, :] = self._console_col.bgr

        # Copy image over
        resized = task.img.resize(ImgSize(w=self._out_img_rect.w, h=self._out_img_rect.h))
        self._out_img_rect.slice_array(self._out.bgr())[:, :, :] = resized.bgr()

        # Now draw the console text if needed:
        console_texts = task.console_texts
        if console_texts is not None:
            if self._min_console_ppn == 0:
                raise RuntimeError("Can't draw console text unless you set console_ppn > 0")
            # Write the console out
            console_line_top_left = self._out_console_rect.p0.copy()
            console_line_top_left.y += 2
            for ct in console_texts:
                text_bbox = self._out.draw.text(
                    text=f">> {ct.text}",
                    font=self._console_font,
                    height=ct.font_height,
                    col=ct.col,
                    origin=console_line_top_left,
                    word_wrap_width=self._out_console_rect.w,
                )
                # Add height to get next position, but 10% of text height for spacing
                console_line_top_left.y += int(1.1 * text_bbox.h)
                # TODO: check if o.y > img height i.e. too much text

        cv2.imshow(self._window_name, self._out.bgr())
        k = cv2.waitKey(task.delay_ms) & 0xFF
        if self._keyboard_callback is not None:
            close = self._keyboard_callback(k)
            if close:
                return True
        return False

    def _run(self):
        self._running = True
        try:
            first = True
            setup = False
            while self._running:
                task: _DrawTask = self._q.get()
                # TODO: errors if q is getting too long

                # Setup on first run (don't do it before, as sometimes there can be a big delay before first frame comes
                # which means you've got an empty screen for a while)
                if first:
                    self._setup_window(task.img.isize)
                    first = False
                    setup = True

                # Stop or draw:
                finish = False
                if not task.stop:
                    finish = self._threaded_draw(task)
                if task.stop or finish:
                    if setup:
                        cv2.destroyWindow(self._window_name)
                    break
        except:  # NOQA
            logger.exception("Failed while drawing!")

        self._running = False
        self._closed.set()

    def draw(self, img: Img, delay_ms: float = 0, console_text: Union[str, List[str]] = None) -> bool:
        """
        Return true if GUI is closed (because of error or keyboard interaction).
        """
        if self._closed.is_set():
            return True
        console_text = console_text if isinstance(console_text, Iterable) else [console_text]
        task = _DrawTask(stop=False, img=img, delay_ms=delay_ms, console_texts=console_text)
        self._q.put(task)
        return False

    def open(self):
        logger.info("Starting OpenCVGUI")
        self._thread = threading.Thread(target=self._run, args=())
        self._thread.daemon = True
        self._thread.start()

    def close(self, timeout: float = None):
        # Clear the queue - don't show what's in the queue, assume the user wants it stopped now now:
        self._q.empty()
        # Add a 'stop' item to the queue - this is our way of telling the _run function to stop processing. (It's a bit
        # of a hack for now to stop the _run method hanging on self._q.get(). TODO make this nicer = ) )
        self._q.put(_DrawTask(stop=True, img=None, console_texts=None, delay_ms=None))
        # Now wait for any processing to have finished (and the above item to be processed) - this avoids us closing
        # while the thread is still running. Timeout, just in case something goes wrong.
        is_set = self._closed.wait(timeout=timeout)
        if not is_set:
            raise RuntimeError(f"Didn't close within {timeout} seconds!")

    def is_alive(self) -> bool:
        if self._thread is None:
            raise RuntimeError("Thread isn't initialized!")
        return self._thread.is_alive()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args, **kwargs):
        self.close()


if __name__ == "__main__":
    import sys

    logger.enable("awareutils")
    with OpenCVGUI(
        "main",
        min_console_ppn=0.3,
        padding_col=Col(30, 30, 30),
        console_col=Col.named.black,
        console_font=cv2.FONT_HERSHEY_DUPLEX,
    ) as gui:
        finished = False
        while not finished:
            img = Img.from_bgr(np.random.randint(low=0, high=255, size=(1080, 1920, 3), dtype=np.uint8))
            finished = gui.draw(
                img,
                delay_ms=1,
                console_text=[
                    ConsoleText(text="the quick brown fox 0123456789"),
                    ConsoleText(text=f"{time.time()}"),
                    ConsoleText(text="look\n^newline character!"),
                    ConsoleText(text="look this line is really long so it'll automatically be word wrapped!"),
                ],
            )
            time.sleep(0.1)
