import pytest
import os
import sys
import traceback
import signal
from loguru import logger
import typing as ty
import pyrootutils
import threading

from chess.engine import SimpleEngine

# Using pyrootutils, we find the root directory of this project and make sure it is our working directory
root = pyrootutils.setup_root(
    search_from=".",
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
    cwd=True,
)


@pytest.fixture(scope="session")
def engine_executable() -> ty.Union[str, ty.List[str]]:
    """Chess engine executable path or command."""
    return os.environ.get("CHESS_ENGINE_EXECUTABLE", "stockfish")


@pytest.fixture(scope="session")
def engine(engine_executable: str) -> SimpleEngine:
    """Chess engine."""
    engine = SimpleEngine.popen_uci(engine_executable)
    return engine


@pytest.fixture(scope="session", autouse=True)
def run_before_and_after_tests(engine: SimpleEngine) -> ty.Generator:
    """Fixture to execute code before and after tests are run.
    We can use this fixture to set up and tear down stuff.
    """
    # Setup: fill with any logic you want
    logger.info("Setting up...")
    # Return to the tests
    with engine:
        yield  # this is where the testing happens
    # Teardown: fill with any logic you want
    _show_threads()
    # _close_all_threads()
    # Finished
    logger.debug("Exiting...")


def _show_threads() -> None:
    """Get a list of all currently alive threads."""
    all_threads = threading.enumerate()
    logger.debug(f"Active threads: {threading.active_count()}")
    for _, thread in enumerate(all_threads):
        logger.debug(f"Thread ({type(thread)}): id={thread.ident}; name={thread.name}")


def _close_all_threads() -> None:
    """Close all threads that are not main. By forcibly killing children threads of the main thread, we can exit PyTest, without altering whatever outcome PyTest was about to return. This is true of course because we do this at the end of the tests."""
    kill_threads = [thread for thread in threading.enumerate() if thread != threading.main_thread()]
    for thread in kill_threads:
        get_thread_source(thread)
        # SimpleEngine
        if isinstance(thread, SimpleEngine) or thread.name == "SimpleEngine":
            # thread.stop()
            logger.info(thread.__dict__)
            logger.trace(f"Killed {thread}")
            # Thread may be dead already, so we try-catch
            try:
                logger.debug(f"Trying to close {thread}")
                signal.pthread_kill(thread.ident, signal.SIGTSTP)  # type: ignore
                logger.trace(f"Killed {thread}")
            except ProcessLookupError as ex:
                logger.warning(ex)
            return


def get_thread_source(thread: threading.Thread) -> ty.Optional[ty.Tuple[str, str, str, str]]:
    """Get info about thread. This will also print the stack trace, so we can see what it is."""
    frame = sys._current_frames().get(thread.ident, None)  # type: ignore
    if frame:
        filename = str(frame.f_code.co_filename)
        name = str(frame.f_code.co_name)
        firstlineno = str(frame.f_code.co_firstlineno)
        source = "".join(traceback.format_stack(frame))
        logger.trace(
            f"Thread={thread};\nFilename={filename}; name={name}; firstlineno={firstlineno};\n\n{source}"
        )
        return filename, name, firstlineno, source
    return None
