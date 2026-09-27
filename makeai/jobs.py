"""Background jobs (dataset processing, tokenizer training, import/export, evaluation) with progress."""
from __future__ import annotations

import threading
import time
import traceback
from typing import Any, Callable

from . import store


class Job:
    def __init__(self, kind: str, title: str):
        self.id = store.new_id("job")
        self.kind = kind
        self.title = title
        self.state = "running"
        self.progress = ""
        self.result: Any = None
        self.error: str | None = None
        self.log: list[str] = []
        self.started = time.time()
        self.ended: float | None = None
        self.cancel = threading.Event()

    def say(self, msg: str) -> None:
        self.progress = msg
        self.log.append(f"[{time.strftime('%H:%M:%S')}] {msg}")
        del self.log[:-500]

    def to_dict(self, full: bool = False) -> dict[str, Any]:
        d = {"id": self.id, "kind": self.kind, "title": self.title, "state": self.state, "progress": self.progress,
             "error": self.error, "started": self.started, "ended": self.ended}
        if full:
            d["result"] = self.result
            d["log"] = self.log
        return d


class Jobs:
    def __init__(self):
        self.jobs: dict[str, Job] = {}

    def start(self, kind: str, title: str, fn: Callable[[Job], Any]) -> Job:
        job = Job(kind, title)
        self.jobs[job.id] = job

        def run():
            try:
                job.result = fn(job)
                job.state = "cancelled" if job.cancel.is_set() else "done"
            except Exception as e:
                job.state = "failed"
                job.error = str(e) or type(e).__name__
                job.log.append(traceback.format_exc())
            finally:
                job.ended = time.time()
        threading.Thread(target=run, name=f"job-{kind}", daemon=True).start()
        return job

    def get(self, job_id: str) -> Job:
        if job_id not in self.jobs:
            raise KeyError(job_id)
        return self.jobs[job_id]

    def list(self) -> list[dict[str, Any]]:
        return [j.to_dict() for j in sorted(self.jobs.values(), key=lambda j: -j.started)][:100]
