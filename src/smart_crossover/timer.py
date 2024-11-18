"""A module of timer."""

import datetime


class Timer:
    """A class to record the time duration of a process.

    Attributes:
        start: The start time of the process.
        end: The end time of the process.
        total_duration: The total duration of the process.

    """
    start: datetime.datetime
    end: datetime.datetime
    total_duration: datetime.timedelta

    def __init__(self) -> None:
        self.start = datetime.datetime.min
        self.end = datetime.datetime.min
        self.total_duration = datetime.timedelta(seconds=0)

    def start_timer(self) -> None:
        """Record the start time of the process."""
        self.start = datetime.datetime.now()

    def end_timer(self) -> None:
        """Record the end time and accumulate the total duration."""
        self.end = datetime.datetime.now()
        self.total_duration += self.end - self.start

    def accumulate_time(self, new_duration: datetime.timedelta) -> None:
        """Add new duration to the total duration."""
        self.total_duration += new_duration

    def clear(self) -> None:
        """Reset the timer."""
        self.__init__()
