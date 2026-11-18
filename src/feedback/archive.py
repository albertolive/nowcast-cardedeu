"""Read forecasts as issued, without reconstructing weather from later observations."""
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import subprocess


def iter_snapshots(since: datetime, until: datetime, *, ref: str = "HEAD",
                   repo: str | Path = ".", legacy_wall_time: bool = False) -> Iterator[dict]:
    """Yield one forecast per issue time, in order, in [since, until).

    Only the legacy parquet repair may request naive keys. Never replay those
    records against a UTC gauge or infer their offsets from a publication date.
    """
    if ((since.tzinfo is None) != legacy_wall_time or
            (until.tzinfo is None) != legacy_wall_time or since >= until):
        raise ValueError("Invalid archive interval or timezone convention")
    # Commit time only narrows the search. A delayed publication can arrive after
    # `until`; it is the timestamp inside the blob that identifies the forecast.
    search_start = since.replace(tzinfo=timezone.utc) if legacy_wall_time else since
    history = subprocess.check_output([
        "git", "log", "--first-parent", "--reverse", "--raw", "--format=",
        "--no-abbrev", "--diff-filter=AM", f"--since={(search_start - timedelta(days=1)).isoformat()}",
        "--end-of-options", ref, "--", "data/latest_prediction.json",
    ], cwd=repo, text=True)
    blobs: dict[datetime, bytes] = {}
    with subprocess.Popen(["git", "cat-file", "--batch"], cwd=repo,
                          stdin=subprocess.PIPE, stdout=subprocess.PIPE) as git:
        assert git.stdin is not None and git.stdout is not None
        try:
            for change in history.splitlines():
                if not change:
                    continue
                oid = change.split()[3]
                git.stdin.write(oid.encode("ascii") + b"\n")
                git.stdin.flush()
                header = git.stdout.readline().split()
                if len(header) != 3 or header[1] != b"blob":
                    raise ValueError(f"Missing archived forecast blob {oid}")
                raw = git.stdout.read(int(header[2]))
                if git.stdout.read(1) != b"\n":
                    raise ValueError(f"Incomplete archived forecast blob {oid}")
                snapshot = json.loads(raw)
                issued = datetime.fromisoformat(snapshot["timestamp"])
                if (issued.tzinfo is None) != legacy_wall_time:
                    continue
                if not legacy_wall_time:
                    issued = issued.astimezone(timezone.utc)
                if since <= issued < until:
                    blobs.setdefault(issued, raw)
        finally:
            git.stdin.close()
        if git.wait():
            raise RuntimeError("git cat-file failed while reading archived forecasts")
    for issued in sorted(blobs):
        yield json.loads(blobs.pop(issued))
