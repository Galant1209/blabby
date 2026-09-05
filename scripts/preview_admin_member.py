"""Loopback-only Admin preview with synthetic in-memory API responses.

Run: python3 scripts/preview_admin_member.py
Open: http://127.0.0.1:8765/admin.html
No credentials, real API, or backend process is used.
"""
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        path = self.path.split('?')[0]
        files = {
            '/admin-member.css': ROOT / 'frontend/app/admin-member.css',
            '/task1-chart-renderer.js': ROOT / 'frontend/app/task1-chart-renderer.js',
        }
        if path in ('/', '/admin.html'):
            body = (ROOT / 'frontend/app/admin.html').read_text()
            body = re.sub(r'<script src="(?:https://[^\"]+|\./config.js)"></script>', '', body)
            body = re.sub(r'<link[^>]+(?:fonts.googleapis|fonts.gstatic)[^>]*>', '', body)
            fixture = (ROOT / 'backend/tests/admin_member_fixture.js').read_text()
            body = body.replace('<script>\n(() => {', '<script>' + fixture + '\ninstallAdminFixture();</script>\n<script>\n(() => {')
            mime = 'text/html; charset=utf-8'
        elif path in files:
            body = files[path].read_text()
            mime = 'text/css' if path.endswith('.css') else 'text/javascript'
        else:
            self.send_error(404); return
        self.send_response(200)
        self.send_header('Content-Type', mime)
        self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'none'; font-src 'none'")
        self.end_headers()
        self.wfile.write(body.encode())

if __name__ == '__main__':
    print('Synthetic Admin preview: http://127.0.0.1:8765/admin.html', flush=True)
    HTTPServer(('127.0.0.1',8765), Handler).serve_forever()
