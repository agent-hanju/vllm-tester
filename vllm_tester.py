#!/usr/bin/env python3
"""
vLLM Tester Server
=======================
HTML 페이지를 직접 서빙하면서 vllm 요청을 실제 vLLM 서버로 프록시하여 CORS를 회피하는 서버 스크립트
Python 표준 라이브러리만 사용 → 폐쇄망 PC 에서도 추가 설치 없이 실행 가능 (3.8+).

사용법:
    python3 vllm_tester.py
        → 기본값: --port 8080, --target http://localhost:8000

    python3 vllm_tester.py --port 9000 --target http://10.0.0.5:8000
        → 9000 포트에서 listen, 10.0.0.5:8000 으로 프록시

    python3 vllm_tester.py --port 8080 --target http://localhost:8000 --api-key sk-xxx
        → /v1/models 조회할 때 위 API key 를 사용 (vLLM 이 --api-key 로 보호 중일 때)

시작 시 동작:
    1. target 의 /v1/models 와 /version 을 호출해서 모델 목록 및 버전 조회
    2. /v1/models 조회 실패하면 즉시 종료 (target URL / vLLM 상태 확인 후 재실행할 것)
    3. HTML 안의 placeholder 를 실제 설정 (target, models, version) 으로 치환해서 캐싱
    4. HTTP 서버 시작

라우팅:
    GET /                  → HTML 페이지
    GET /v1/models         → 프록시 (브라우저 측에서 직접 부를 수도 있게)
    POST /v1/chat/completions → 프록시
    그 외 /v1/*            → 프록시
    POST /tokenize         → 프록시
    POST /detokenize       → 프록시
"""

import argparse
import http.client
import http.server
import json
import logging
import os
import socketserver
import sys
import urllib.parse
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='vLLM Chat Tester Server (proxy + HTML 서빙)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--port', type=int, default=8080,
                   help='이 서버가 listen 할 포트 (default: 8080)')
    p.add_argument('--target', default='http://localhost:8000',
                   help='vLLM base URL. /v1 은 붙이지 말 것 (default: http://localhost:8000)')
    p.add_argument('--api-key', default=None,
                   help='vLLM 이 --api-key 로 보호된 경우, /v1/models 조회용 키')
    p.add_argument('--html', default=None,
                   help='HTML 파일 경로 (default: 같은 디렉토리의 vllm-tester.html)')
    p.add_argument('--bind', default='',
                   help="bind 주소. 기본값은 모든 인터페이스 (''). "
                        "본인만 쓸 거면 '127.0.0.1' 추천")
    p.add_argument('--timeout', type=int, default=600,
                   help='upstream timeout (초). 시작 시 조회 및 프록시 공통 적용 (default: 600)')
    p.add_argument('--log-level', default='INFO',
                   choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                   help='로그 레벨 (default: INFO)')
    args = p.parse_args()

    args.target = args.target.rstrip('/')
    if args.target.endswith('/v1'):
        logger.warning("--target 끝의 '/v1' 제거: %s → %s", args.target, args.target[:-3])
        args.target = args.target[:-3]

    if args.html is None:
        here = os.path.dirname(os.path.abspath(__file__))
        args.html = os.path.join(here, 'vllm-tester.html')

    return args

# ============================================================================
# vLLM 정보 조회 (시작 시 한 번)
# ============================================================================

def _make_conn(
    scheme: str, host: str, port: int, timeout: int
) -> http.client.HTTPConnection:
    if scheme == 'https':
        return http.client.HTTPSConnection(host, port, timeout=timeout)
    return http.client.HTTPConnection(host, port, timeout=timeout)


def _parse_target(target_url: str) -> Tuple[str, str, int]:
    """target_url → (scheme, host, port). host 가 없으면 RuntimeError."""
    parsed = urllib.parse.urlparse(target_url)
    if not parsed.hostname:
        raise RuntimeError(f"--target URL 에서 호스트를 파싱할 수 없습니다: {target_url!r}")
    scheme = parsed.scheme
    host: str = parsed.hostname
    port: int = parsed.port or (443 if scheme == 'https' else 80)
    return scheme, host, port


def _get(
    target_url: str,
    path: str,
    api_key: Optional[str] = None,
    timeout: int = 30,
) -> Tuple[int, bytes]:
    scheme, host, port = _parse_target(target_url)
    headers: Dict[str, str] = {}
    if api_key:
        headers['Authorization'] = 'Bearer ' + api_key
    conn = _make_conn(scheme, host, port, timeout)
    try:
        conn.request('GET', path, headers=headers)
        resp = conn.getresponse()
        return resp.status, resp.read()
    finally:
        conn.close()



def fetch_models(
    target_url: str, api_key: Optional[str] = None, timeout: int = 30
) -> List[str]:
    """vLLM 의 /v1/models 호출 → 모델 ID 리스트 반환. 실패 시 RuntimeError."""
    status, body = _get(target_url, '/v1/models', api_key=api_key, timeout=timeout)
    if status != 200:
        raise RuntimeError(
            f"GET {target_url}/v1/models → {status}\n"
            f"  응답: {body.decode('utf-8', errors='replace')[:500]}"
        )
    try:
        data: Dict[str, Any] = json.loads(body)
    except Exception as e:
        raise RuntimeError(f"/v1/models 응답이 JSON 이 아님: {e}") from e

    items = data.get('data', [])
    if not isinstance(items, list):
        raise RuntimeError(
            f"/v1/models 응답에 'data' 배열이 없음. 받은 키: {list(data.keys())}"
        )

    models: List[str] = []
    for item in items: # pyright: ignore[reportUnknownVariableType]
        if isinstance(item, dict) and 'id' in item and isinstance(item['id'], str):
            models.append(item['id'])
    if not models:
        raise RuntimeError("/v1/models 응답에 모델이 없습니다 (data 가 빈 배열)")
    return models


def fetch_version(
    target_url: str, api_key: Optional[str] = None, timeout: int = 30
) -> str:
    """vLLM 의 /version 호출 → 버전 문자열 반환. 실패 시 'unknown' 반환."""
    try:
        status, body = _get(target_url, '/version', api_key=api_key, timeout=timeout)
        if status != 200:
            return 'unknown'
        data: Dict[str, object] = json.loads(body)
        version = data.get('version')
        return version if isinstance(version, str) else 'unknown'
    except Exception:
        return 'unknown'


# ============================================================================
# HTTP 핸들러 (HTML 서빙 + /v1/* 프록시)
# ============================================================================

# 응답에서 다시 보내면 안 되는 hop-by-hop 헤더 (RFC 7230 §6.1)
HOP_BY_HOP = {
    'connection', 'keep-alive', 'proxy-authenticate', 'proxy-authorization',
    'te', 'trailers', 'transfer-encoding', 'upgrade',
}

_PROXY_EXTRA = frozenset({'/tokenize', '/detokenize'})
_CHUNK_SIZE = 1024


def make_handler(target_url: str, html_bytes: bytes, timeout: int = 600) -> type:
    """target / HTML 을 캡처한 BaseHTTPRequestHandler 서브클래스를 만들어 반환."""
    scheme, target_host, target_port = _parse_target(target_url)
    target_https: bool = (scheme == 'https')

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            logger.debug(format, *args)

        def _serve_html(self) -> None:
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(html_bytes)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(html_bytes)

        def _proxy(self) -> None:
            length = int(self.headers.get('Content-Length', 0))
            body: Optional[bytes] = self.rfile.read(length) if length else None

            headers: Dict[str, str] = {}
            for k, v in self.headers.items():
                kl = k.lower()
                if kl in HOP_BY_HOP or kl in ('host', 'content-length'):
                    continue
                headers[k] = v
            headers['Host'] = f"{target_host}:{target_port}"

            try:
                conn = _make_conn(
                    'https' if target_https else 'http',
                    target_host, target_port, timeout=timeout,
                )
                conn.request(self.command, self.path, body, headers)
                resp = conn.getresponse()
            except Exception as e:
                logger.error("upstream 연결 실패 [%s %s]: %s", self.command, self.path, e)
                self.send_response(502)
                self.send_header('Content-Type', 'text/plain; charset=utf-8')
                self.end_headers()
                self.wfile.write(f"Upstream error: {e}\n".encode('utf-8'))
                return

            self.send_response(resp.status, resp.reason)
            for k, v in resp.getheaders():
                if k.lower() in HOP_BY_HOP:
                    continue
                self.send_header(k, v)
            self.end_headers()

            try:
                while True:
                    chunk = resp.read(_CHUNK_SIZE)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass  # 클라이언트 측이 먼저 끊은 경우
            finally:
                conn.close()

        def _is_proxy_path(self, path: str) -> bool:
            return path.startswith('/v1/') or path in _PROXY_EXTRA

        def do_GET(self) -> None:
            path = self.path.split('?', 1)[0]
            if path in ('/', '/index.html'):
                self._serve_html()
            elif self._is_proxy_path(path):
                self._proxy()
            else:
                self.send_response(404)
                self.send_header('Content-Type', 'text/plain; charset=utf-8')
                self.end_headers()
                self.wfile.write(b'Not Found\n')

        def do_POST(self) -> None:
            path = self.path.split('?', 1)[0]
            if self._is_proxy_path(path):
                self._proxy()
            else:
                self.send_response(404)
                self.end_headers()

        def do_PUT(self) -> None:
            path = self.path.split('?', 1)[0]
            if self._is_proxy_path(path):
                self._proxy()
            else:
                self.send_response(404)
                self.end_headers()

        def do_DELETE(self) -> None:
            path = self.path.split('?', 1)[0]
            if self._is_proxy_path(path):
                self._proxy()
            else:
                self.send_response(404)
                self.end_headers()

    return Handler


class ThreadingServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True

# ============================================================================
# 메인
# ============================================================================

def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
    )

    if not os.path.exists(args.html):
        logger.error("HTML 파일 없음: %s", args.html)
        logger.error("--html 로 경로를 지정하거나, vllm-tester.html 을 같은 디렉토리에 두세요.")  # noqa: E501
        sys.exit(1)
    with open(args.html, 'r', encoding='utf-8') as f:
        html_template = f.read()

    logger.info("target: %s", args.target)
    logger.info("/v1/models 조회 중...")

    try:
        models = fetch_models(args.target, api_key=args.api_key, timeout=args.timeout)
    except Exception as e:
        logger.error("모델 조회 실패: %s", e)
        logger.error("  • --target 가 올바른지 확인 (현재: %s)", args.target)
        logger.error("  • vLLM 서버가 떠있는지 확인")
        logger.error("  • vLLM 이 --api-key 로 보호 중이면 --api-key 옵션도 추가")
        sys.exit(1)

    logger.info("모델 %d개 확인:", len(models))
    for m in models:
        logger.info("  • %s", m)

    logger.info("/version 조회 중...")
    version = fetch_version(args.target, api_key=args.api_key, timeout=args.timeout)
    logger.info("vLLM version: %s", version)

    config_json = json.dumps(
        {'target': args.target, 'models': models, 'version': version},
        ensure_ascii=False,
    )
    if '__SERVER_CONFIG_JSON__' not in html_template:
        logger.error("HTML 파일에 '__SERVER_CONFIG_JSON__' placeholder 가 없음.")
        logger.error("서버 모드 전용 HTML 이 맞는지 확인하세요.")
        sys.exit(1)
    html_rendered = html_template.replace('__SERVER_CONFIG_JSON__', config_json)
    html_bytes = html_rendered.encode('utf-8')

    bind_display = args.bind if args.bind else 'localhost'
    logger.info("listen: http://%s:%d  (Ctrl+C 로 종료)", bind_display, args.port)

    handler_cls = make_handler(args.target, html_bytes, timeout=args.timeout)
    try:
        ThreadingServer((args.bind, args.port), handler_cls).serve_forever()
    except KeyboardInterrupt:
        logger.info("종료")


if __name__ == '__main__':
    main()
