import json
import subprocess
import time
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def save_result(kind: str, payload: dict) -> Path:
    """Every run is kept as eval/results/<kind>_<timestamp>.json, so numbers in the README can be traced."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = RESULTS / f"{kind}_{stamp}.json"
    payload = {"kind": kind, "created": stamp, "git_commit": git_commit(), **payload}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def fmt_ci(ci: dict, digits: int = 3) -> str:
    return f"{ci['mean']:.{digits}f} [{ci['lo']:.{digits}f}, {ci['hi']:.{digits}f}]"
