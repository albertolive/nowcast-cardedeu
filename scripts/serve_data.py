#!/usr/bin/env python3
"""Servidor HTTP lleuger amb CORS per servir dades de predicció al dashboard."""
import http.server
import os
import sys

ALLOWED_FILES = {"latest_prediction.json", "predictions_log.jsonl"}


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
        super().do_GET()

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.end_headers()

    def log_message(self, format, *args):
        pass


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 80
    directory = sys.argv[2] if len(sys.argv) > 2 else "/app/data"
    os.chdir(directory)
    server = http.server.HTTPServer(("0.0.0.0", port), DataHandler)
    print(f"📡 Data server on :{port} (serving {directory})")
    server.serve_forever()
