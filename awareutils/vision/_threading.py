from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from queue import Empty, Queue
from threading import Event, Thread
from typing import Any

from loguru import logger

"""
A threading pattern we commonly use, inc. a few little helpers.
"""


@dataclass
class HeavyTaskQueueItem:
    stop_run_loop: bool
    data: Any
    processed: Event


class ProcessHeavyTaskInBackground(metaclass=ABCMeta):
    def __init__(self, max_task_queue_size: int = -1, max_result_queue_size: int = -1):
        self._thread = None
        self._stopping = False
        self._setup_complete = Event()
        self._event_run_loop_started = Event()
        self._event_run_loop_finished = Event()
        self._task_queue = Queue(max_task_queue_size)
        self._result_queue = Queue(max_result_queue_size)
        self._opened = False
        self._closed = False

    @abstractmethod
    def setup_for_heavy_tasks(self) -> None:
        pass

    @abstractmethod
    def run_heavy_task(self, task: Any, idx: int) -> None:
        pass

    @abstractmethod
    def cleanup_from_heavy_tasks(self, did_setup) -> None:
        pass

    def add_next_task_in_background(self, data: Any = None, block: bool = False):
        """
        Wait until the current one has finished processing, and then add a new one.
        """
        if self._stopping:
            logger.warning("Can't add more items now we've stopped")
            return
        if self._task_queue.full():
            logger.warning(
                (
                    "Task queue is full so will block - consider adding tasks slower, or having a larger"
                    " max_task_queue_size"
                )
            )
        task = HeavyTaskQueueItem(stop_run_loop=False, data=data, processed=Event())
        self._task_queue.put(task)
        if block:
            task.processed.wait()

    def run(self) -> None:
        self._event_run_loop_started.set()
        try:
            # Setup:
            try:
                self.setup_for_heavy_tasks()
                self._setup_complete.set()
            except:  # NOQA
                logger.exception("User-defined setup_for_heavy_tasks failed")
                self._stopping = True
            # Now run it
            idx = -1
            while not self._stopping:
                idx += 1
                task = self._task_queue.get()

                # Run/stop:
                if not task.stop_run_loop:
                    try:
                        result = self.run_heavy_task(task.data, idx=idx)
                        if self._result_queue.full():
                            logger.warning(
                                (
                                    "Result queue is full so will block - consider adding tasks slower, or having a"
                                    " larger max_task_queue_size"
                                )
                            )
                        self._result_queue.put(result)
                    except StopIteration:
                        # If this is raised in the heavy task, we stop
                        logger.debug("StopIteration raised - bailing")
                        self._stopping = True
                    except:  # NOQA
                        logger.exception("Failed while running user-defined run_heavy_task")
                        self._stopping = True

                    task.processed.set()
                else:
                    self._stopping = True

                if self._stopping:
                    logger.debug("Cleaning up from heavy tasks ...")
                    self.cleanup_from_heavy_tasks(did_setup=self._setup_complete.is_set())
        except:  # NOQA
            logger.exception("Failed running run loop")

        # Clear the buffer:
        while not self._task_queue.empty():
            task = self._task_queue.get()
            if not task.stop_run_loop:
                task.processed.set()

        logger.debug("Thread loop finished")
        self._event_run_loop_finished.set()

    def is_alive(self) -> bool:
        if self._thread is None:
            raise RuntimeError("Thread isn't initialized!")
        return self._thread.is_alive()

    def open(self) -> None:
        if self._opened:
            raise RuntimeError(f"Can only open {self.__class__} once!")
        logger.info("Starting thread for {cls}", cls=self.__class__.__name__)
        self._thread = Thread(target=self.run, args=())
        self._thread.daemon = True
        self._thread.start()
        self._opened = True

    def close(self, timeout: float = None, force: bool = False) -> None:
        if self._event_run_loop_finished.is_set():
            return
        # Send a close task to the run loop - the user is responsible for cleaning anything up. For example, in OpenCV
        # everything needs to happen in the same thread, so we need to do the closing there.
        self._task_queue.put(HeavyTaskQueueItem(stop_run_loop=True, data=None, processed=None))
        if not self._event_run_loop_finished.wait(timeout=timeout):
            raise RuntimeError(f"Didn't close within {timeout} seconds!")
        self._closed = True

    def consume_results(self, timeout: float = 5):
        # self._block_until(self._event_run_loop_iteration_result_ready, timeout)
        while True:
            try:
                result = self._result_queue.get(block=True, timeout=timeout)
            except Empty:
                raise RuntimeError(f"No new result appears in {timeout} seconds")
            yield result

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args, **kwargs):
        self.close()

    @staticmethod
    def _block_until(event, timeout=None):
        if not event.wait(timeout=timeout):
            raise ValueError(f"Couldn't get event within {timeout} seconds!")
