import time
import datetime


def loginfo(msg: str):
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}")


class MeasureTimer:
    """
    A simple timer to measure durations of code segments (tick -> tock).
    Once it has collected 'counts_log' samples, it logs min, max, and average
    and resets its internal samples list.
    """

    def __init__(self, counts_log=50, label="timer"):
        """
        Args:
            counts_log (int): number of samples to collect before logging stats
            label (str): an optional name/label to include in logs
        """
        self.counts_log = counts_log
        self.label = label

        self._start_t = None
        self._durations = []

    def tick(self):
        """
        Record the current time as start time.
        """
        self._start_t = time.time()

    def tock(self):
        """
        Finish measuring time. Add the duration to our list of samples.
        If samples have reached 'counts_log', print min, max, average, and reset.
        """
        if self._start_t is None:
            loginfo(f"[WARN][{self.label}] 'tock()' called without 'tick()'; ignoring.")
            return

        duration = time.time() - self._start_t
        self._durations.append(duration)
        self._start_t = None  # reset start

        if len(self._durations) >= self.counts_log:
            dur_min = min(self._durations)
            dur_max = max(self._durations)
            dur_avg = sum(self._durations) / len(self._durations)
            loginfo(
                f"[{self.label}] timing over last {self.counts_log} calls: "
                f"min={dur_min:.4f}s, max={dur_max:.4f}s, avg={dur_avg:.4f}s"
            )
            self._durations.clear()
