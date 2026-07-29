#
#  Copyright (C) 2017-2025 Dremio Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
"""Helpers for Dynamic Client Registration proxying."""

import json
import time

from aiohttp import ClientError, ClientSession
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from dremioai import log

logger = log.logger(__name__)

_REGISTER_ALLOWED_FIELDS = {"redirect_uris", "client_name", "scope"}
_HOP_BY_HOP_HEADERS = {
    "connection",
    "content-encoding",
    "content-length",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}
_UPSTREAM_TIMEOUT_SECONDS = 30
_MAX_LOG_BODY_BYTES = 8192


def _truncate_bytes(body: bytes | None) -> str | None:
    if not body:
        return None
    return body[:_MAX_LOG_BODY_BYTES].decode("utf-8", errors="replace")


def _normalize_register_request(
    request_headers: dict[str, str], request_body: bytes
) -> tuple[dict[str, str], bytes]:
    if not request_body:
        return request_headers, request_body

    content_type = request_headers.get("content-type", "")
    if "application/json" not in content_type.lower():
        return request_headers, request_body

    try:
        payload = json.loads(request_body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return request_headers, request_body

    if not isinstance(payload, dict):
        return request_headers, request_body

    normalized_payload = {
        key: value for key, value in payload.items() if key in _REGISTER_ALLOWED_FIELDS
    }
    normalized_body = json.dumps(normalized_payload, separators=(",", ":")).encode(
        "utf-8"
    )
    return request_headers, normalized_body


def _response_headers(headers) -> dict[str, str]:
    return {
        key: value
        for key, value in headers.items()
        if key.lower() not in _HOP_BY_HOP_HEADERS
    }


def _proxy_error_response(request: Request, upstream_url: str, error: str) -> JSONResponse:
    logger.warning(
        "OAuth register proxy upstream failed",
        method=request.method,
        local_path=request.url.path,
        upstream_url=upstream_url,
        status_code=502,
        error=error,
    )
    return JSONResponse(
        {
            "error": "oauth_register_upstream_unavailable",
            "message": error,
            "upstream_url": upstream_url,
            "method": request.method,
            "path": request.url.path,
        },
        status_code=502,
    )


def augment_register_response(
    response_body: bytes, content_type: str | None
) -> tuple[bytes, bool]:
    if not content_type or "application/json" not in content_type.lower():
        return response_body, False

    try:
        payload = json.loads(response_body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return response_body, False

    if not isinstance(payload, dict):
        return response_body, False

    changed = False
    if "client_id_issued_at" not in payload:
        payload["client_id_issued_at"] = int(time.time())
        changed = True
    if "application_type" not in payload:
        payload["application_type"] = "web"
        changed = True
    if "client_secret_expires_at" not in payload:
        payload["client_secret_expires_at"] = 0
        changed = True

    if not changed:
        return response_body, False

    return json.dumps(payload, separators=(",", ":")).encode("utf-8"), True


async def proxy_register_request(request: Request, upstream_url: str) -> Response:
    request_body = await request.body()
    request_headers = {
        key: value
        for key, value in request.headers.items()
        if key.lower() not in {"host", "content-length"}
    }
    request_headers, request_body = _normalize_register_request(
        request_headers, request_body
    )

    logger.info(
        "OAuth register proxy request",
        method=request.method,
        local_path=request.url.path,
        upstream_url=upstream_url,
        query_string=str(request.url.query),
        headers=request_headers,
        body=_truncate_bytes(request_body),
        body_truncated=bool(request_body and len(request_body) > _MAX_LOG_BODY_BYTES),
    )

    try:
        async with ClientSession() as session:
            async with session.request(
                request.method,
                upstream_url,
                params=request.query_params.multi_items(),
                data=request_body or None,
                headers=request_headers,
                allow_redirects=False,
                timeout=_UPSTREAM_TIMEOUT_SECONDS,
            ) as upstream_response:
                response_body = await upstream_response.read()
                response_headers = _response_headers(upstream_response.headers)
                if 200 <= upstream_response.status < 300:
                    response_body, _ = augment_register_response(
                        response_body, upstream_response.content_type
                    )

                logger.info(
                    "OAuth register proxy response",
                    method=request.method,
                    local_path=request.url.path,
                    upstream_url=upstream_url,
                    status_code=upstream_response.status,
                    headers=response_headers,
                    body=_truncate_bytes(response_body),
                    body_truncated=bool(
                        response_body and len(response_body) > _MAX_LOG_BODY_BYTES
                    ),
                )

                return Response(
                    content=response_body,
                    status_code=upstream_response.status,
                    headers=response_headers,
                    media_type=upstream_response.content_type,
                )
    except ClientError as exc:
        return _proxy_error_response(
            request,
            upstream_url,
            f"Failed to reach upstream auth endpoint: {exc}",
        )
