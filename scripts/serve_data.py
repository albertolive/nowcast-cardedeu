#!/usr/bin/env python3
"""Servidor HTTP lleuger amb CORS per servir dades de predicció al dashboard."""
import http.server
import json
import os
import sys

ALLOWED_FILES = {"latest_prediction.json", "predictions_log.jsonl"}

# Only these fields are needed by the dashboard history view.
# The full file (~15 MB) is kept on disk for training; we serve a slim version.
_DASHBOARD_FIELDS = {
    "timestamp", "probability_pct", "rain_category",
    "verified", "actual_rain", "actual_rain_mm", "correct",
}

# Cache the stripped JSONL to avoid reprocessing on every request.
_jsonl_cache: dict = {"mtime": 0.0, "data": b""}


def _get_slim_jsonl(path: str) -> bytes:
    """Return predictions_log.jsonl with only dashboard-needed fields, cached by mtime."""
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return b""
    if mtime == _jsonl_cache["mtime"]:
        return _jsonl_cache["data"]

    lines = []
    with open(path) as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
                slim = {k: obj[k] for k in _DASHBOARD_FIELDS if k in obj}
                lines.append(json.dumps(slim, separators=(",", ":")))
            except json.JSONDecodeError:
                continue
    data = ("\n".join(lines) + "\n").encode() if lines else b""
    _jsonl_cache["mtime"] = mtime
    _jsonl_cache["data"] = data
    return data


class DataHandler(http.server.SimpleHTTPRequestHandler):
    """Serves only allowed data files with CORS headers."""

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        super().end_headers()

    def do_GET(self):
        filename = self.path.lstrip("/").split("?")[0]
        if filename not in ALLOWED_FILES:
            self.send_error(404, "Not Found")
            return
        if filename == "predictions_log.jsonl":
            data = _get_slim_jsonl(filename)
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        super().do_GET()

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.end_headers()

    def log_message(self, format, *args):
        pass


class ReusableHTTPServer(http.server.HTTPServer):
    # Allow fast rebind so a restart doesn't race on TIME_WAIT sockets.
    allow_reuse_address = True


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 80
    directory = sys.argv[2] if len(sys.argv) > 2 else "/app/data"
    os.chdir(directory)
    server = ReusableHTTPServer(("0.0.0.0", port), DataHandler)
    print(f"📡 Data server on :{port} (serving {directory})")
    server.serve_forever()
