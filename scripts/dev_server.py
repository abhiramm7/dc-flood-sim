#!/usr/bin/env python3
"""Static dev server for docs/ that disables HTTP caching.

Plain `python -m http.server` sends Last-Modified but no Cache-Control,
so browsers heuristically cache index.html and keep loading old module
versions after edits. This wrapper sends no-cache on everything.

Usage: python3 scripts/dev_server.py [port]   (default 8901)
"""
import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer


class NoCacheHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Cache-Control', 'no-cache, must-revalidate')
        super().end_headers()

    def log_message(self, *args):
        pass


if __name__ == '__main__':
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8901
    handler = partial(NoCacheHandler, directory='docs')
    print(f'serving docs/ on http://127.0.0.1:{port}/ (no-cache)')
    ThreadingHTTPServer(('', port), handler).serve_forever()
