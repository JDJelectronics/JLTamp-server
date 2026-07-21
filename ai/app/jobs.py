"""Async job tracking for prompt requests.

The app fires a prompt and then polls, so a request must return immediately
with an id. This keeps that contract (aiService.ts expects `job_id`,
`poll_interval` and `timeout`) with the bookkeeping tightened up:

* A job that outlives its budget is reported as timed out. Python cannot kill
  the worker thread, so the thread is left to finish and its result discarded —
  but the caller is told the truth instead of polling forever.
* The duplicate guard and the job table are updated together under one lock,
  so a crash cannot leave a prompt permanently marked "already running".
"""
from __future__ import annotations

import threading
import time
import uuid
from typing import Callable


class JobManager:
    def __init__(self, max_jobs: int = 20, timeout: int = 120):
        self.max_jobs = max_jobs
        self.timeout = timeout
        self._lock = threading.Lock()
        self._jobs: dict[str, dict] = {}
        self._active: dict[str, str] = {}     # prompt -> job_id
        threading.Thread(target=self._reaper, daemon=True).start()

    # ── submission ───────────────────────────────────────────────────────────
    def existing(self, prompt: str) -> str | None:
        with self._lock:
            job_id = self._active.get(prompt)
            if job_id and self._jobs.get(job_id, {}).get("status") in ("queued", "processing"):
                return job_id
            # Stale entry: the job finished or vanished.
            if job_id:
                self._active.pop(prompt, None)
            return None

    def full(self) -> bool:
        with self._lock:
            live = sum(1 for j in self._jobs.values()
                       if j.get("status") in ("queued", "processing"))
            return live >= self.max_jobs

    def submit(self, prompt: str, work: Callable[[str], dict]) -> str:
        job_id = uuid.uuid4().hex[:16]
        now = time.time()
        with self._lock:
            self._jobs[job_id] = {
                "status": "queued",
                "prompt": prompt,
                "progress": 0,
                "step": "In de wachtrij...",
                "eta": 30.0,
                "created_at": now,
            }
            self._active[prompt] = job_id
        threading.Thread(target=self._run, args=(job_id, prompt, work),
                         daemon=True).start()
        return job_id

    # ── execution ────────────────────────────────────────────────────────────
    def _run(self, job_id: str, prompt: str, work: Callable[[str], dict]) -> None:
        start = time.time()
        self._patch(job_id, {"status": "processing", "progress": 10,
                             "step": "Prompt analyseren...", "started_at": start})
        try:
            result = work(prompt)
            duration = time.time() - start
            if self._expired(job_id):
                # Already reported as timed out; publishing now would contradict
                # what the client was told.
                return
            failed = result.get("status") == "error"
            self._replace(job_id, {
                "status": "error" if failed else "done",
                "result": None if failed else result,
                "error": result.get("message") if failed else None,
                "progress": 100,
                "eta": 0,
                "duration": round(duration, 2),
                "finished_at": time.time(),
            })
        except Exception as e:                      # noqa: BLE001
            self._replace(job_id, {
                "status": "error",
                "error": str(e),
                "progress": 100,
                "eta": 0,
                "duration": round(time.time() - start, 2),
                "finished_at": time.time(),
            })
        finally:
            with self._lock:
                if self._active.get(prompt) == job_id:
                    self._active.pop(prompt, None)

    def _expired(self, job_id: str) -> bool:
        with self._lock:
            return self._jobs.get(job_id, {}).get("status") == "timeout"

    # ── state ────────────────────────────────────────────────────────────────
    def _patch(self, job_id: str, patch: dict) -> None:
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].update(patch)

    def _replace(self, job_id: str, state: dict) -> None:
        with self._lock:
            if job_id in self._jobs:
                state["prompt"] = self._jobs[job_id].get("prompt")
                self._jobs[job_id] = state

    def get(self, job_id: str) -> dict | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return None
            out = dict(job)
        # A live job's ETA is only meaningful relative to now.
        if out.get("status") == "processing":
            elapsed = time.time() - out.get("started_at", time.time())
            out["eta"] = round(max(0.0, self.timeout - elapsed), 1)
        out.pop("prompt", None)
        return out

    def _reaper(self) -> None:
        """Time out overrunning jobs and drop finished ones after a grace period."""
        while True:
            time.sleep(5)
            now = time.time()
            with self._lock:
                for job_id, job in list(self._jobs.items()):
                    status = job.get("status")
                    if status in ("queued", "processing"):
                        started = job.get("started_at") or job.get("created_at", now)
                        if now - started > self.timeout:
                            prompt = job.get("prompt")
                            self._jobs[job_id] = {
                                "status": "timeout",
                                "error": f"job liep langer dan {self.timeout}s",
                                "progress": 100,
                                "eta": 0,
                                "finished_at": now,
                                "prompt": prompt,
                            }
                            if prompt and self._active.get(prompt) == job_id:
                                self._active.pop(prompt, None)
                    elif job.get("finished_at") and now - job["finished_at"] > 600:
                        del self._jobs[job_id]

    def stats(self) -> dict:
        with self._lock:
            live = sum(1 for j in self._jobs.values()
                       if j.get("status") in ("queued", "processing"))
            return {"total": len(self._jobs), "active": live}
