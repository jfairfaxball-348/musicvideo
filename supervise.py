"""Keep the serial renderer running across transient process and driver failures."""

import json
import msvcrt
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
LOCAL = ROOT / ".local"
MANIFEST = ROOT / "output/manifest.json"


def current():
    return json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}


def renderer_running():
    with (LOCAL / "worker.lock").open("ab+") as f:
        f.seek(0)
        try:
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            return True
        f.seek(0)
        msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
        return False


def monitor():
    attempts = 0
    while attempts < 3:
        while renderer_running():
            time.sleep(30)
        state = current()
        if (LOCAL / "STOP").exists():
            return
        if (
            state.get("status") == "validated"
            and (ROOT / "output/LNMA_Live_Rap_Metal_Official_Music_Video.mp4").exists()
        ):
            return
        count = len(state.get("completed", {}))
        result = subprocess.run([sys.executable, "-u", "render.py"], cwd=ROOT)
        if result.returncode == 0:
            return
        attempts = 1 if len(current().get("completed", {})) > count else attempts + 1
        print(
            f"Worker exited {result.returncode}; consecutive failures at current position: {attempts}",
            flush=True,
        )
        time.sleep(15 * attempts)
    state = current()
    state["status"] = "failed; see worker error log"
    tmp = MANIFEST.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(MANIFEST)
    raise SystemExit(1)


if __name__ == "__main__":
    os.chdir(ROOT)
    with (LOCAL / "supervisor.lock").open("ab+") as lock:
        lock.seek(0)
        try:
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            sys.exit(0)
        monitor()
