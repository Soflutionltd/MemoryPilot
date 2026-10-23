import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
import unittest

from memorypilot import MemoryPilot, MemoryPilotError


class _Handler(BaseHTTPRequestHandler):
    seen: list

    def log_message(self, *_args):
        return

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length).decode()
        self.seen.append(
            {
                "path": self.path,
                "authorization": self.headers.get("Authorization"),
                "subject": self.headers.get("X-MemoryPilot-Subject"),
                "body": body,
            }
        )
        if self.path == "/v1/memories":
            self._reply(201, {"memory": {"id": "m1"}, "merged": False})
            return
        self._reply(403, {"error": "This key lacks the 'read' permission"})

    def _reply(self, status, payload):
        encoded = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


class ClientTest(unittest.TestCase):
    def test_add_memory_and_errors(self):
        _Handler.seen = []
        server = HTTPServer(("127.0.0.1", 0), _Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        host, port = server.server_address
        client = MemoryPilot(f"http://{host}:{port}", "mpk_test").for_subject("alice")
        created = client.add_memory(content="Alice prefers green tea")
        self.assertEqual(created["memory"]["id"], "m1")
        self.assertEqual(_Handler.seen[0]["authorization"], "Bearer mpk_test")
        self.assertEqual(_Handler.seen[0]["subject"], "alice")
        with self.assertRaises(MemoryPilotError) as raised:
            client.search("tea")
        self.assertEqual(raised.exception.status, 403)
        server.shutdown()


if __name__ == "__main__":
    unittest.main()
