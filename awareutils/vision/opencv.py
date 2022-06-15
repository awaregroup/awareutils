import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, List, Tuple, Union

import cv2
import numpy as np
from awareutils.vision._threading import Threadable
from awareutils.vision.col import Col
from awareutils.vision.img import Img, ImgSize
from awareutils.vision.shape import Pixel, Rectangle
from awareutils.vision.video import ThreadedOpenCVVideoWriter
from loguru import logger


@dataclass
class ConsoleText:
    text: str
    font_height: float = 0.015
    col: Col = Col.named.white


@dataclass
class _DrawTask:
    img: Img
    console_texts: List[ConsoleText] = field(default_factory=list)
    delay_ms: float = 1


def default_keyboard_callback(self: "OpenCVGUI", key: int) -> bool:
    # Return whether or not to finish
    if chr(key) == "q":
        logger.info("'q' pressed - closing")
        return True
    return False


class OpenCVGUI(Threadable):
    """
    A GUI built on OpenCV primitives that's:
        a) A little easier to use with common nice things (like a text 'console' area etc.), and
        b) Threaded, so the drawing computation happens in a separate thread.
    """

    def __init__(
        self,
        window_name: str = None,
        record_path: Path = None,
        record_fps: Path = 30,
        keyboard_callback: Callable = None,
        mouse_callback: Callable = None,
        min_console_ppn: float = 0,
        console_font: int = cv2.FONT_HERSHEY_PLAIN,
        console_col: Col = Col.named.black,
        padding_col: Col = Col(30, 30, 30),
    ):
        """
        If min_console_ppn is > 0 it means we want to have an area beside (not over top of) the image for putting
        random text information. If the image is too wide for the screen, then the console will be at the bottom
        (instead of padding top and bottom) ... with a minimum size of min_console_ppn of the height. Likewise if the
        image is too high, we pad on the sides, etc.
        """
        super().__init__()

        self._window_name = "OpenCVGUI" if window_name is None else window_name
        self._record_path = record_path
        self._record_fps = record_fps
        self._mouse_callback = mouse_callback
        self._keyboard_callback = default_keyboard_callback if keyboard_callback is None else keyboard_callback
        self._window_setup = False
        self._window_size: ImgSize = None
        self._min_console_ppn = min_console_ppn
        self._console_col = console_col
        self._console_font = console_font
        self._padding_col = padding_col
        self._out: Img = None
        self._out_img_rect: Rectangle = None
        self._out_console_rect: Rectangle = None
        self._vo = None
        self.last_mouse_x: int = None
        self.last_mouse_y: int = None
        self.last_mouse_event = None
        self._isize: ImgSize = None

    def _on_mouse(self, event, x, y, *args, **kwargs):
        self.last_mouse_x = int(x / self._window_size.w * self._isize.w)
        self.last_mouse_y = int(y / self._window_size.h * self._isize.h)
        self.last_mouse_event = event
        if self._mouse_callback is not None:
            return self._mouse_callback(self, event, x, y)

    def setup_in_thread(self):
        pass

    def _setup_window(self, isize: ImgSize):

        # Persist this size (we'll check all match it later)
        self._isize = isize

        # TODO: allow non fullscreen
        # The "right" way to do fullscreen windows:
        cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(self._window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.moveWindow(self._window_name, 0, 0)
        # Register our callback
        cv2.setMouseCallback(self._window_name, self._on_mouse)

        # Get aspect ratio and calculate padding we need, inc. console area
        _, _, window_w, window_h = cv2.getWindowImageRect(self._window_name)
        self._window_size = ImgSize(w=window_w, h=window_h)
        logger.debug("Window size: {window_size}", window_size=self._window_size)
        logger.debug("Image size: {img_size}", img_size=self._isize)
        window_aspect_ratio = self._window_size.aspect_ratio
        img_aspect_ratio = self._isize.aspect_ratio

        # # If we're adding a console, we just pretend the image is bigger by that much
        # if self._min_console_ppn > 0:
        #     img_size = ImgSize(w=int(img_size.w * (1 + self._min_console_ppn)), h=img_size.h)
        #     img_aspect_ratio = img_size.w / img_size.h
        # Now do the padding.
        if img_aspect_ratio >= window_aspect_ratio:
            # Img is wider than the screen, so pad top and bottom
            drawn_h = int(self._window_size.w / img_aspect_ratio)
            if self._min_console_ppn > 0:
                # If we're doing a console, draw the image at the top, and claim everything underneath as console.
                pad_ppn = (self._window_size.h - drawn_h) / self._window_size.h
                img_x0 = 0
                img_x1 = self._window_size.w - 1
                if pad_ppn < self._min_console_ppn:
                    # OK, got to bump it so we get more space. This also means we need to reduce the size of the image
                    # in width too ...
                    new_drawn_h = self._window_size.h * (1 - self._min_console_ppn)
                    new_drawn_w = new_drawn_h / drawn_h * self._window_size.w
                    img_x0 = int((self._window_size.w - new_drawn_w) / 2)
                    img_x1 = img_x0 + new_drawn_w - 1
                    drawn_h = new_drawn_h
                drawn_img_rect = Rectangle(
                    p0=Pixel(x=img_x0, y=0, isize=self._window_size),
                    p1=Pixel(x=img_x1, y=drawn_h - 1, isize=self._window_size),
                )
                console_rect = Rectangle(
                    p0=Pixel(x=0, y=drawn_h, isize=self._window_size),
                    p1=Pixel(x=self._window_size.w - 1, y=self._window_size.h - 1, isize=self._window_size),
                )
            else:
                pad_top = int((self._window_size.h - drawn_h) / 2)
                drawn_img_rect = Rectangle(
                    p0=Pixel(x=0, y=pad_top, isize=self._window_size),
                    p1=Pixel(x=self._window_size.w - 1, y=pad_top + drawn_h - 1, isize=self._window_size),
                )
                console_rect = None
        else:
            # Img is narrower than screen, so pad sides.
            drawn_w = int(self._window_size.h * img_aspect_ratio)
            if self._min_console_ppn > 0:
                # If we're adding a console, put the image on the left and claim everything to the right as console
                pad_ppn = (self._window_size.w - drawn_w) / self._window_size.w
                img_y0 = 0
                img_y1 = self._window_size.h - 1
                if pad_ppn < self._min_console_ppn:
                    # OK, got to bump it so we get more space. This also means we need to reduce the size of the image
                    # in height too ...
                    new_drawn_w = self._window_size.w * (1 - self._min_console_ppn)
                    new_drawn_h = new_drawn_w / drawn_w * self._window_size.h
                    img_y0 = int((self._window_size.h - new_drawn_h) / 2)
                    img_y1 = img_y0 + new_drawn_h - 1
                    drawn_w = new_drawn_w

                drawn_img_rect = Rectangle(
                    p0=Pixel(x=0, y=img_y0, isize=self._window_size),
                    p1=Pixel(x=drawn_w - 1, y=img_y1, isize=self._window_size),
                )
                console_rect = Rectangle(
                    p0=Pixel(x=drawn_w, y=0, isize=self._window_size),
                    p1=Pixel(x=self._window_size.w - 1, y=self._window_size.h - 1, isize=self._window_size),
                )
            else:
                pad_left = int((self._window_size.w - drawn_w) / 2)
                drawn_img_rect = Rectangle(
                    p0=Pixel(x=pad_left, y=0, isize=self._window_size),
                    p1=Pixel(x=pad_left + drawn_w - 1, y=self._window_size.h - 1, isize=self._window_size),
                )
                console_rect = None

        logger.debug("drawn_img_rect: {drawn_img_rect}", drawn_img_rect=drawn_img_rect)
        logger.debug("console_rect: {console_rect}", console_rect=console_rect)

        # Save things
        self._out = Img.new_bgr(self._window_size, col=self._padding_col)
        self._out_img_rect = drawn_img_rect
        self._out_console_rect = console_rect
        self._window_setup = True

        # Set up video recorder:
        if self._record_path is not None:
            self._vo = ThreadedOpenCVVideoWriter(
                self._record_path, height=window_h, width=window_w, fps=self._record_fps
            )
            self._vo.open()

    def run_task_in_thread(self, task: _DrawTask, idx: int) -> Tuple[bool, Any]:
        """
        Return true if we should quit the drawing loop (based on keyboard interaction)
        """

        # Set up if we need to:
        if not self._window_setup:
            self._setup_window(isize=task.img.isize)

        # Check the img size hasn't changed ... this doesn't really matter as things get resized, except it'll mess up
        # mouse coordinates (which assume img isize)
        if task.img.isize != self._isize:
            raise RuntimeError(
                f"GUI was set up with size {self._isize} so can only take that size, but you passed {task.img.isize}"
            )

        # Clear out console:
        if self._out_console_rect is not None:
            self._out_console_rect.slice_array(self._out.bgr())[:, :, :] = self._console_col.bgr

        # Copy image over
        resized = task.img.resize(ImgSize(w=self._out_img_rect.w, h=self._out_img_rect.h))
        self._out_img_rect.slice_array(self._out.bgr())[:, :, :] = resized.bgr()
        # Now draw the console text if needed:
        console_texts = task.console_texts
        if console_texts is not None and console_texts:
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

        # Tell it we're done
        if self._keyboard_callback is not None:
            close = self._keyboard_callback(self, k)
            if close:
                return True, None

        # Save if needed:
        if self._vo is not None:
            # Write a copy so self._out doesn't get overwritten
            self._vo.write(self._out.copy())

        return False, None

    def teardown_in_thread(self, did_setup) -> None:
        if did_setup:
            cv2.destroyWindow(self._window_name)
            if self._vo is not None:
                self._vo.close()

    def draw(
        self,
        img: Img,
        delay_ms: float = 0,
        console_text: Union[ConsoleText, List[ConsoleText]] = None,
        # block: bool = False,
    ) -> bool:
        """
        Return true if GUI is closed (because of error or keyboard interaction). The block argument is used for
        scenarios where we might be calling draw in a loop very fast so don't want it to run constantly - so it waits
        on the draw task to be complete (including the waitKey delay) before continuing. (NB: this largely defeats the
        point of threading.) In other cases where the loop is 'slow', e.g. we're drawing as fast as we can read a frame
        from a video, it's fine to call it non-blocking.
        """
        if console_text is not None:
            if not isinstance(console_text, Iterable):
                console_text = [console_text]
            for t in console_text:
                if not isinstance(t, ConsoleText):
                    raise RuntimeError(f"console_text must be ConsoleText not {type(t)}")
        result = self.add_next_task_and_get_result_of_previous(
            _DrawTask(img=img, delay_ms=delay_ms, console_texts=console_text)
        )
        return result.stopped


if __name__ == "__main__":

    logger.enable("awareutils")
    with OpenCVGUI(
        min_console_ppn=0.3,
        padding_col=Col(30, 30, 30),
        console_col=Col.named.black,
        console_font=cv2.FONT_HERSHEY_DUPLEX,
    ) as gui:
        while True:
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
            if finished:
                break
