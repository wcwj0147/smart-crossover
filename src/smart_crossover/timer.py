"""A module of timer."""

import datetime


class Timer:
    start: datetime.datetime
    end: datetime.datetime
    total_duration: datetime.timedelta

    def __init__(self) -> None:
        self.start = datetime.datetime.min
        self.end = datetime.datetime.min
        self.total_duration = datetime.timedelta(seconds=0)

    def start_timer(self) -> None:
        self.start = datetime.datetime.now()

    def end_timer(self) -> None:
        self.end = datetime.datetime.now()
        self.total_duration += self.end - self.start

    def accumulate_time(self, new_duration: datetime.timedelta) -> None:
        self.total_duration += new_duration

    def clear(self) -> None:
        self.__init__()
