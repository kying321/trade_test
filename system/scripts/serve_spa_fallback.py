#!/usr/bin/env python3
from __future__ import annotations

import argparse
import http.server
from pathlib import Path
from urllib.parse import unquote, urlsplit


def should_fallback_to_index(dist_dir: Path, request_path: str) -> bool:
    raw_path = urlsplit(str(request_path or "")).path or "/"
    normalized = unquote(raw_path).strip() or "/"
    if normalized == "/":
        return False
    requested = dist_dir / normalized.lstrip("/")
    if requested.exists():
        return False
    if requested.suffix:
        return False
    return True


def build_handler(dist_dir: Path):
    class SpaFallbackHandler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(dist_dir), **kwargs)

        def do_GET(self):  # noqa: N802
            if should_fallback_to_index(dist_dir, self.path):
                original_path = self.path
                self.path = "/index.html"
                try:
                    return super().do_GET()
                finally:
                    self.path = original_path
            return super().do_GET()

    return SpaFallbackHandler


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve a static SPA dist directory with index.html fallback for deep routes.")
    parser.add_argument("port", type=int)
    parser.add_argument("host")
    parser.add_argument("dist_dir")
    args = parser.parse_args()

    dist_dir = Path(args.dist_dir).expanduser().resolve()
    handler = build_handler(dist_dir)
    with http.server.ThreadingHTTPServer((args.host, args.port), handler) as server:
        server.serve_forever()


if __name__ == "__main__":
    main()
