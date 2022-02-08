import threading
from abc import ABCMeta, abstractmethod

from loguru import logger

"""
A threading pattern we commonly use, inc. a few little helpers.
"""


class Threadable(metaclass=ABCMeta):
    def __init__(self):
        self._thread = None
        self._thread_finished = threading.Event()
        self._opened = False

    def open(self) -> None:
        if self._opened:
            raise RuntimeError(f"Can only open {self.__class__} once!")
        logger.info("Starting thread for {cls}", cls=self.__class__.__name__)
        self._thread = threading.Thread(target=self.run, args=())
        self._thread.daemon = True
        self._thread.start()
        self._opened = True

    @abstractmethod
    def _close_immediately(self) -> None:
        pass

    @abstractmethod
    def _close_after_thread_finished(self) -> None:
        pass

    def close(self, timeout: float = None) -> None:
        self._close_immediately()
        is_set = self._thread_finished.wait(timeout=timeout)
        if not is_set:
            raise RuntimeError(f"Didn't close within {timeout} seconds!")
        self._close_after_thread_finished()

    @abstractmethod
    def _run(self) -> None:
        pass

    def run(self) -> None:
        try:
            self._run()
        except:  # NOQA
            logger.exception(f"Failed while running thread {self.__class__.__name__}")
        self._thread_finished.set()

    def is_alive(self) -> bool:
        if self._thread is None:
            raise RuntimeError("Thread isn't initialized!")
        return self._thread.is_alive()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args, **kwargs):
        self.close()

    @staticmethod
    def _block_until(event, timeout=None):
        is_set = event.wait(timeout=timeout)
        if not is_set:
            raise ValueError(f"Couldn't get event within {timeout} seconds!")
