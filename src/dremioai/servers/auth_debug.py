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
"""OAuth debugging proxy routes for ``--auth-debug`` mode."""

import json
from typing import Any

from aiohttp import ClientError, ClientSession
from mcp.server.auth.json_response import PydanticJSONResponse
from pydantic import AnyHttpUrl
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from dremioai import log
from dremioai.api.oauth_metadata import OAuthMetadataRFC8414
from dremioai.servers.mcp import (
    build_authorization_server_metadata,
    build_protected_resource_metadata,
    request_base_url,
)

logger = log.logger(__name__)

_UPSTREAM_TIMEOUT_SECONDS = 30
_MAX_LOG_BODY_BYTES = 8192
_REGISTER_ALLOWED_FIELDS = {
    "client_name",
    "redirect_uris",
    "grant_types",
    "response_types",
    "scope",
    "token_endpoint_auth_method",
}
_HOP_BY_HOP_HEADERS = {
    "connection",
    "content-length",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}


def _truncate_bytes(body: bytes | None) -> str | None:
    if not body:
        return None
    return body[:_MAX_LOG_BODY_BYTES].decode("utf-8", errors="replace")


def _response_headers(headers) -> dict[str, str]:
    return {
        key: value
        for key, value in headers.items()
        if key.lower() not in _HOP_BY_HOP_HEADERS
    }


def _proxy_error_response(request: Request, upstream_url: str, error: str) -> JSONResponse:
    payload = {
        "error": "auth_debug_upstream_unavailable",
        "message": error,
        "upstream_url": upstream_url,
        "method": request.method,
        "path": request.url.path,
    }
    logger.warning(
        "Auth debug proxy upstream failed",
        method=request.method,
        local_path=request.url.path,
        upstream_url=upstream_url,
        status_code=502,
        error=error,
    )
    return JSONResponse(payload, status_code=502)


def _normalize_register_request(
    request_headers: dict[str, str], request_body: bytes
) -> tuple[dict[str, str], bytes, dict[str, Any] | None, list[str]]:
    if not request_body:
        return request_headers, request_body, None, []

    content_type = request_headers.get("content-type", "")
    if "application/json" not in content_type.lower():
        return request_headers, request_body, None, []

    try:
        payload = json.loads(request_body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return request_headers, request_body, None, []

    if not isinstance(payload, dict):
        return request_headers, request_body, None, []

    removed_fields = sorted(
        key for key in payload.keys() if key not in _REGISTER_ALLOWED_FIELDS
    )
    if not removed_fields:
        return request_headers, request_body, payload, []

    normalized_payload = {
        key: value for key, value in payload.items() if key in _REGISTER_ALLOWED_FIELDS
    }
    normalized_body = json.dumps(normalized_payload, separators=(",", ":")).encode(
        "utf-8"
    )
    return request_headers, normalized_body, payload, removed_fields


async def _proxy_request(request: Request, upstream_url: str) -> Response:
    request_body = await request.body()
    request_headers = {
        key: value
        for key, value in request.headers.items()
        if key.lower() not in {"host", "content-length"}
    }
    original_json_body = None
    removed_fields: list[str] = []

    if request.url.path == "/oauth/register":
        request_headers, request_body, original_json_body, removed_fields = (
            _normalize_register_request(request_headers, request_body)
        )
        if removed_fields:
            logger.info(
                "Auth debug normalized register request",
                local_path=request.url.path,
                removed_fields=removed_fields,
                original_body=original_json_body,
                forwarded_body=_truncate_bytes(request_body),
            )

    logger.info(
        "Auth debug proxy request",
        method=request.method,
        local_path=request.url.path,
        upstream_url=upstream_url,
        query_string=str(request.url.query),
        headers=request_headers,
        body=_truncate_bytes(request_body),
        body_truncated=bool(request_body and len(request_body) > _MAX_LOG_BODY_BYTES),
        normalized_register_fields_removed=removed_fields or None,
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

                logger.info(
                    "Auth debug proxy response",
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

def _build_local_authorization_server_metadata(
    request: Request, auth_metadata: OAuthMetadataRFC8414
) -> OAuthMetadataRFC8414:
    local_base = request_base_url(request)
    return OAuthMetadataRFC8414(
        issuer=AnyHttpUrl(local_base),
        authorization_endpoint=f"{local_base}/oauth/authorize",
        token_endpoint=f"{local_base}/oauth/token",
        registration_endpoint=AnyHttpUrl(f"{local_base}/oauth/register"),
        scopes_supported=auth_metadata.scopes_supported,
        response_types_supported=auth_metadata.response_types_supported,
        grant_types_supported=auth_metadata.grant_types_supported,
        code_challenge_methods_supported=auth_metadata.code_challenge_methods_supported,
        token_endpoint_auth_methods_supported=auth_metadata.token_endpoint_auth_methods_supported,
    )


def register_auth_debug_routes(mcp) -> None:
    """Register OAuth discovery and proxy routes for auth debugging."""
    auth_metadata = build_authorization_server_metadata()
    if auth_metadata is None:
        raise ValueError("--auth-debug requires OAuth to be configured")

    upstream_authorize_url = str(auth_metadata.authorization_endpoint)
    upstream_token_url = str(auth_metadata.token_endpoint)
    upstream_register_url = str(auth_metadata.registration_endpoint)

    logger.info(
        "Registering auth debug routes",
        upstream_authorize_url=upstream_authorize_url,
        upstream_token_url=upstream_token_url,
        upstream_register_url=upstream_register_url,
    )

    @mcp.custom_route("/oauth/register", methods=["POST"])
    async def _register(request: Request) -> Response:
        return await _proxy_request(request, upstream_register_url)

    @mcp.custom_route("/oauth/authorize", methods=["GET"])
    async def _authorize(request: Request) -> Response:
        return await _proxy_request(request, upstream_authorize_url)

    @mcp.custom_route("/oauth/token", methods=["POST"])
    async def _token(request: Request) -> Response:
        return await _proxy_request(request, upstream_token_url)

    @mcp.custom_route("/.well-known/oauth-protected-resource", methods=["GET"])
    @mcp.custom_route(
        "/.well-known/oauth-protected-resource/{resource_path:path}",
        methods=["GET"],
    )
    async def _protected_resource_metadata(
        request: Request, resource_path: str = ""
    ) -> Response:
        metadata = build_protected_resource_metadata(request, resource_path)
        logger.info(
            "Auth debug protected resource metadata",
            path=request.url.path,
            resource=metadata.resource,
            authorization_servers=metadata.authorization_servers,
        )
        return PydanticJSONResponse(metadata)

    @mcp.custom_route("/.well-known/oauth-authorization-server", methods=["GET"])
    @mcp.custom_route("/mcp/.well-known/oauth-authorization-server", methods=["GET"])
    @mcp.custom_route(
        "/mcp/{project_id}/.well-known/oauth-authorization-server", methods=["GET"]
    )
    async def _metadata(request: Request) -> Response:
        metadata = _build_local_authorization_server_metadata(request, auth_metadata)
        logger.info(
            "Auth debug metadata served locally",
            local_path=request.url.path,
            issuer=metadata.issuer,
            authorization_endpoint=metadata.authorization_endpoint,
            token_endpoint=metadata.token_endpoint,
            registration_endpoint=metadata.registration_endpoint,
        )
        return PydanticJSONResponse(metadata)
