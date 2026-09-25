#!/usr/bin/env python3
"""Local HTTP failure fixture; no external service or real credential is used."""
import http.server
import subprocess
import sys
import threading
import time

paths = []


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        paths.append(self.path)
        self.rfile.read(int(self.headers.get('Content-Length', '0')))
        if self.headers.get('Authorization') is not None:
            self.send_error(401)
            return
        if self.path == '/slow':
            time.sleep(.5)
        if self.path == '/error':
            self.send_response(503)
        elif self.path == '/redirect':
            self.send_response(302)
            self.send_header('Location', '/must-not-follow')
        else:
            self.send_response(200)
        self.end_headers()
        try:
            self.wfile.write(b'x' * 40000 if self.path == '/large' else b'{}')
        except (BrokenPipeError, ConnectionResetError):
            pass


server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
thread = threading.Thread(target=server.serve_forever)
thread.start()
try:
    result = subprocess.run([sys.argv[1], f'http://127.0.0.1:{server.server_port}'],
                            timeout=10)
    assert '/must-not-follow' not in paths
    raise SystemExit(result.returncode)
finally:
    server.shutdown()
    server.server_close()
    thread.join()
