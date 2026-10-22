"""tqdm that also writes its state to outputs/progress.json, read by the Claude Code status line.

Usage: `from progress import track` then `for x in track(examples, "lora"): ...`
"""
import json
import os
import time
from pathlib import Path

from tqdm import tqdm

PATH = Path(__file__).parent / "outputs" / "progress.json"


def _write(bar, done=False):
    d = bar.format_dict
    state = dict(desc=bar.desc, n=d["n"], total=d["total"], rate=d["rate"] or (d["n"] / d["elapsed"] if d["elapsed"] else None),
                 elapsed=d["elapsed"], updated=time.time(), done=done, pid=os.getpid())
    PATH.parent.mkdir(exist_ok=True)
    tmp = PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(state))
    tmp.replace(PATH)  # atomic, so a reader never sees half a file


def track(iterable, desc, total=None, every=10.0):
    """Iterate like tqdm, rewriting the progress file at most every `every` seconds."""
    bar = tqdm(iterable, desc=desc, total=total, mininterval=every)
    last = 0.0
    for x in bar:
        yield x
        if time.time() - last > every:
            _write(bar)
            last = time.time()
    _write(bar, done=True)
