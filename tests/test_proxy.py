import base64
import hashlib
import http.client
import http.server
import json
import os
import threading
import unittest
from typing import ClassVar, Dict, Optional

from server import ThreadingServer, make_handler


class RecordingUpstream(http.server.BaseHTTPRequestHandler):
    received_body: ClassVar[Optional[bytes]] = None
    received_headers: ClassVar[Dict[str, str]] = {}
    response_body: ClassVar[bytes] = b'{"ok":true}'

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        return

    def do_POST(self) -> None:
        length = int(self.headers.get('Content-Length', '0'))
        type(self).received_body = self.rfile.read(length)
        type(self).received_headers = dict(self.headers.items())
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(type(self).response_body)))
        self.end_headers()
        self.wfile.write(type(self).response_body)


class ProxyBodyTest(unittest.TestCase):
    upstream: http.server.ThreadingHTTPServer
    proxy: ThreadingServer
    upstream_thread: threading.Thread
    proxy_thread: threading.Thread

    @classmethod
    def setUpClass(cls) -> None:
        cls.upstream = http.server.ThreadingHTTPServer(('127.0.0.1', 0), RecordingUpstream)
        cls.upstream_thread = threading.Thread(target=cls.upstream.serve_forever, daemon=True)
        cls.upstream_thread.start()

        upstream_port = cls.upstream.server_address[1]
        handler = make_handler(
            f'http://127.0.0.1:{upstream_port}',
            b'<html></html>',
            static_dir=os.getcwd(),
            timeout=30,
        )
        cls.proxy = ThreadingServer(('127.0.0.1', 0), handler)
        cls.proxy_thread = threading.Thread(target=cls.proxy.serve_forever, daemon=True)
        cls.proxy_thread.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.proxy.shutdown()
        cls.proxy.server_close()
        cls.upstream.shutdown()
        cls.upstream.server_close()
        cls.proxy_thread.join(timeout=5)
        cls.upstream_thread.join(timeout=5)

    def post_through_proxy(self, body: bytes) -> bytes:
        RecordingUpstream.received_body = None
        RecordingUpstream.received_headers = {}
        conn = http.client.HTTPConnection(
            '127.0.0.1', self.proxy.server_address[1], timeout=30)
        try:
            conn.request(
                'POST',
                '/v1/chat/completions',
                body=body,
                headers={'Content-Type': 'application/json'},
            )
            response = conn.getresponse()
            self.assertEqual(response.status, 200)
            return response.read()
        finally:
            conn.close()

    def assert_opaque_round_trip(self, body: bytes) -> None:
        response_body = self.post_through_proxy(body)
        received = RecordingUpstream.received_body
        self.assertIsNotNone(received)
        assert received is not None
        self.assertEqual(len(received), len(body))
        self.assertEqual(
            RecordingUpstream.received_headers.get('Content-Length'), str(len(body)))
        self.assertEqual(hashlib.sha256(received).digest(), hashlib.sha256(body).digest())
        self.assertEqual(received, body)
        self.assertEqual(response_body, RecordingUpstream.response_body)

    def test_canonical_multimodal_json_is_forwarded_unchanged(self) -> None:
        payload = {
            'model': 'test-model',
            'messages': [{
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': 'describe'},
                    {
                        'type': 'image_url',
                        'image_url': {'url': 'data:image/png;base64,iVBORw=='},
                    },
                ],
            }],
        }
        body = json.dumps(payload, ensure_ascii=False, separators=(',', ':')).encode()
        self.assert_opaque_round_trip(body)

    def test_one_megabyte_media_payload_is_forwarded_unchanged(self) -> None:
        raw_media = bytes(range(256)) * 4096
        data_url = 'data:video/mp4;base64,' + base64.b64encode(raw_media).decode('ascii')
        payload = {
            'messages': [{
                'role': 'user',
                'content': [{'type': 'video_url', 'video_url': {'url': data_url}}],
            }],
        }
        body = json.dumps(payload, separators=(',', ':')).encode()
        self.assertGreater(len(body), 1024 * 1024)
        self.assert_opaque_round_trip(body)


if __name__ == '__main__':
    unittest.main()
