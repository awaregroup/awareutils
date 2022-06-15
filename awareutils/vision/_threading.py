from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from threading import Event, Lock, Thread
from typing import Any, Set, Tuple

from loguru import logger

"""
A threading pattern we commonly use, inc. a few little helpers.
"""


class WaitInterruptibleEvent(Event):
    """
    In use, this Event can wait() like a normal class for it's own flag to be set(), but it also waits on the second
    event i.e. triggers until either flag is set. Generally this is used to avoid being stuck waiting forever e.g.
    "wait as normal, unless we want to exit, which we didn't know until after we started waiting".
    """

    def __init__(self):
        super().__init__()
        self._interrupted = False

    def wait(self, clear: bool = False, timeout: float = None) -> Tuple[bool, bool]:
        # A convenience method that waits as normal, but then also clears it (which we always do) and then on return
        # lets us know why it stopped waiting - the interruption or not.
        ret = super().wait(timeout=timeout)
        if clear:
            super().clear()
        self._interrupted = False
        return self._interrupted, ret

    def interupt(self):
        self._interrupted = True
        super().set()


class WaitInterruptingEvent(Event):
    """
    This class will set a bunch of other events when it itself is set. Commonly used as a blanket stopping event so that
    if it's ever set, all others will be, and hence all other waits() are interrupted. A simple way to avoid getting
    stuck waiting forever.
    """

    def __init__(self):
        super().__init__()
        self._events: Set[WaitInterruptibleEvent] = set()

    def set(self):
        """
        This is all it really boils down to - overriding `set` so that it `set`s other events too.
        """
        for e in self._events:
            e.interupt()
        super().set()

    def create_interruptible_event(self) -> WaitInterruptibleEvent:
        event = WaitInterruptibleEvent()
        self._events.add(event)
        return event

    def remove_interruptible_event(self, event: WaitInterruptibleEvent) -> None:
        self._events.remove(event)


@dataclass(frozen=False)
class _Task:
    idx: int
    data: Any
    result: Any
    processed: Event
    run: bool


@dataclass
class _Result:
    idx: int
    result: Any
    stopped: bool
    first: bool


class Threadable(metaclass=ABCMeta):
    """
    Our base thread that:
      - Supports processing a single background task, before being given another.
      - Is "simple" - hopefully less likely to have unexpected thready problems (e.g. race conditions or growing queues
        taking up memory).
      - It supports setup/teardown processes in the main threads, and the run loop tasks easily allowing the thread to
        be stopped.
    The main secret sauce is having custom events we act on, but whose wait can be interrupted by the global '_stopping'
    event. This way, we don't get hung up waiting when we should be closing (and in a more simple fashion than e.g.
    adding a 'stop' task onto the end of a queue and hoping nothing blocks before then).
    """

    def __init__(self):
        self._stopping = WaitInterruptingEvent()
        self._thread = None
        self._next_task_ready = self._stopping.create_interruptible_event()
        self._setup_complete = self._stopping.create_interruptible_event()
        self._run_loop_finished = Event()  # self._stopping.create_interruptible_event()
        self._next_task_to_process: _Task = None
        self._opened = False
        self._closed = False
        self._next_task_setting_lock = Lock()

    @abstractmethod
    def setup_in_thread(self) -> None:
        pass

    @abstractmethod
    def run_task_in_thread(self, task: Any, idx: int) -> Tuple[bool, Any]:
        """
        Returns a tuple where the first item is if the thread should stop, and the second is any result data.
        """
        pass

    @abstractmethod
    def teardown_in_thread(self, did_setup) -> None:
        pass

    def add_next_task_and_get_result_of_previous(self, data: Any = None, timeout: float = None) -> _Result:
        """
        This is our main method of interaction. The intended use case is:

        ```
        # kick things off with the first task
        add_next_task_and_get_result_of_previous()
        while True:
            # Wait until we get the result from the previous task, and when we've got that, add the next task
            stop, result = add_next_task_and_get_result_of_previous()
            if stop:
                break
        ```
        """

        if self._stopping.is_set():
            logger.warning("Can't add more items now we've stopped")
            return _Result(idx=None, first=self._next_task_to_process is None, stopped=True, result=None)

        next_task = _Task(
            idx=0,
            data=data,
            result=None,
            # Don't let stopping interrupt this:
            processed=Event(),  # self._stopping.create_interruptible_event(),
            run=False,
        )

        # If it's our first run, just add it and return:
        if self._next_task_to_process is None:
            with self._next_task_setting_lock:
                self._next_task_to_process = next_task
                # self._next_task_ready.clear()
                self._next_task_ready.set()
            return _Result(idx=0, first=True, stopped=False, result=None)

        # Otherwise, wait until we've finished processing the last one
        current_task = self._next_task_to_process
        assert current_task is not None
        print("waiting for current task to be processed", current_task)
        ret = current_task.processed.wait(timeout=timeout)
        print("done waiting for current task to be processed", current_task)
        if not ret:
            raise RuntimeError("Timed out waiting for previous task to be processed!")

        result = current_task.result
        # self._stopping.remove_interruptible_event(current_task.processed_or_stopped)
        with self._next_task_setting_lock:
            next_task.idx = current_task.idx + 1
            print("adding new task", next_task)
            self._next_task_to_process = next_task
            # self._next_task_ready.clear()
            self._next_task_ready.set()
        return _Result(idx=current_task.idx, first=False, stopped=False, result=result)

    def run(self) -> None:
        try:
            # Setup:
            try:
                logger.info("Setting up in thread")
                self.setup_in_thread()
                self._setup_complete.set()
            except:  # NOQA
                logger.exception("User-defined setup_in_thread failed. Stopping.")
                self._stopping.set()

            # Now run it. Our job here is to ensure that any new task is processed. Even if we close, we want to ensure
            # that the last task added still gets processed.
            while True:  # not self._stopping.is_set():

                # Wait for a new task or for a stopping interrupt:
                print("waiting ...", self._stopping.is_set())
                stopping, _ = self._next_task_ready.wait(clear=True)

                # OK, at this stage, either we have a new task we need to process, or we got told to stop (in which case
                # the task is an old one). Validate that:
                task = self._next_task_to_process
                print("task", task, "<", stopping, self._stopping.is_set())
                if stopping:
                    assert self._stopping.is_set()
                if stopping:  # or self._stopping.is_set():
                    # OK, the current task won't actually be new, as we interrupted before it was ready
                    if not task.run and not task.processed.set():
                        raise RuntimeError("This task should be run")
                else:
                    if task.run or task.processed.is_set():
                        print("hello")
                        print(task.run, task.processed.is_set(), self._stopping.is_set())
                        raise RuntimeError("This task should not already have been run")

                # Process it:
                print(self.__class__.__name__, stopping, task.idx, task.processed.is_set())
                print("xxxx")
                if not task.run:
                    task_says_stop = False
                    try:
                        logger.debug(f"Processing task #{task.idx}")
                        task_says_stop, result = self.run_task_in_thread(task.data, idx=task.idx)
                        task.result = result
                    except:  # NOQA
                        logger.exception("User-defined run_task_in_thread failed. Stopping.")
                        task_says_stop = True
                    # Update our state:
                    task.run = True
                    task.processed.set()

                    # Stop if we need:
                    if task_says_stop:
                        logger.info("User task requested stop")
                        self._stopping.set()

                # Leave the loop when we're done:
                if self._stopping.is_set() and task.run:
                    break

                # # Now, process the task if we need to. We always finish processing all tasks (even if we're stopping)
                # if not stopping:
                #     if task.processed_or_stopped.is_set() and not task.processed_or_stopped._interrupted:
                #         raise RuntimeError(
                #             "This shouldn't happen - the task has already been processed without being interrupted."
                #         )
                #     if task.run:
                #         # We can see this again if a stop is called
                #         stopping = True
                #     else:
                #         try:
                #             logger.info(f"Writing {task.idx}")
                #             stopping, result = self.run_task_in_thread(task.data, idx=idx)
                #             task.result = result
                #             task.processed_or_stopped.set()
                #             task.run = True
                #         except:  # NOQA
                #             logger.exception("User-defined run_task_in_thread failed. Stopping.")
                #             task.processed_or_stopped.set()
                #             stopping = True

                # if stopping or self._stopping.is_set():
                #     self._stopping.set()  # As this wasn't set in the above loop
                #     logger.info("Tearing down in thread")
                #     try:
                #         self.teardown_in_thread(did_setup=self._setup_complete.is_set())
                #     except:  # NOQA
                #         logger.exception("User-defined teardown_in_thread failed. Stopping.")
                #         # Don't need to do anything here, as we're already in the process of stopping

                #     print("BREAKING")
                #     break  # Redundant, but be safe

            # Tear down:
            try:
                logger.info("Tearing down in thread")
                self.teardown_in_thread(did_setup=self._setup_complete.is_set())
            except:  # NOQA
                logger.exception("User-defined teardown_in_thread failed. Stopping.")
                # Don't need to do anything here, as we're already in the process of stopping

        except:  # NOQA
            logger.exception("Failed running run loop")
            self._stopping.set()
            # Unblock any waits on the current task:
            if self._next_task_to_process is not None:
                self._next_task_to_process.processed.set()

        logger.info("Thread loop finished")
        self._run_loop_finished.set()

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

    def close(self, timeout: float = None) -> None:
        # assert False
        if self._closed:
            return
        # task = self._next_task_to_process
        # if task is not None:
        #     logger.info("Waiting for last task to finish before closing")
        #     task.processed.wait(timeout=timeout)
        logger.info("CLOSING")
        self._stopping.set()
        # Wait for the loop to finish (including after tearing down):
        if not self._run_loop_finished.wait(timeout=timeout):
            raise RuntimeError(f"Didn't close within {timeout} seconds!")
        print("closed???")
        self._closed = True

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args, **kwargs):
        self.close()

    def wait_until_setup_complete(self, timeout: float = None):
        stopped, ret = self._setup_complete.wait(clear=False, timeout=timeout)
        if stopped:
            raise RuntimeError("Thread stopped before setup was even completed!")
        if not ret:
            raise RuntimeError("Timed out while waiting for setup to be complete!")
