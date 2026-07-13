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

import contextlib
import socket
import uuid
from typing import AsyncGenerator
from unittest.mock import ANY
from urllib.parse import parse_qs, urlparse

import pytest
from httpx import AsyncClient
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, RedirectResponse
from starlette.routing import Route

from dremioai.config import settings
from dremioai.config.tools import ToolType
from dremioai.servers.mcp import Transports, init
from mocks.http_mock import ServerFixture, start_server_with_app


def _reserve_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        return sock.getsockname()[1]


def _protected_resource_metadata_url(mcp_url: str) -> str:
    parsed = urlparse(mcp_url)
    normalized_path = parsed.path.rstrip("/") or "/"
    return parsed._replace(
        path=f"/.well-known/oauth-protected-resource{normalized_path}"
    ).geturl()


class _UpstreamAuthState:
    def __init__(self):
        self.authorize_requests = []
        self.token_requests = []
        self.register_requests = []


@contextlib.asynccontextmanager
async def auth_debug_server(project_id: str) -> AsyncGenerator[tuple[ServerFixture, _UpstreamAuthState], None]:
    old = settings.instance()
    upstream_fixture = None
    mcp_fixture = None
    try:
        upstream_state = _UpstreamAuthState()
        upstream_port = _reserve_local_port()
        mcp_port = _reserve_local_port()
        while mcp_port == upstream_port:
            mcp_port = _reserve_local_port()

        upstream_base = f"http://127.0.0.1:{upstream_port}"

        async def authorize(request: Request):
            upstream_state.authorize_requests.append(dict(request.query_params))
            redirect_uri = request.query_params["redirect_uri"]
            state = request.query_params.get("state", "")
            sep = "&" if "?" in redirect_uri else "?"
            return RedirectResponse(
                f"{redirect_uri}{sep}code=upstream-code&state={state}",
                status_code=302,
            )

        async def token(request: Request):
            form = dict(await request.form())
            upstream_state.token_requests.append(
                {
                    "form": form,
                    "headers": dict(request.headers),
                }
            )
            return JSONResponse(
                {"proxied": True, "grant_type": form.get("grant_type")},
                status_code=200,
                headers={"x-upstream-auth-debug": "token"},
            )

        async def register(request: Request):
            body = await request.json()
            upstream_state.register_requests.append(
                {
                    "json": body,
                    "headers": dict(request.headers),
                }
            )
            return JSONResponse(
                {"client_id": "upstream-client", "client_name": body.get("client_name")},
                headers={"x-upstream-auth-debug": "register"},
            )

        upstream_app = Starlette(
            routes=[
                Route("/oauth/authorize", authorize, methods=["GET"]),
                Route("/oauth/token", token, methods=["POST"]),
                Route("/oauth/register", register, methods=["POST"]),
            ]
        )
        upstream_server, upstream_stop = start_server_with_app(
            upstream_app,
            host="127.0.0.1",
            port=upstream_port,
            log_level="warning",
            name="upstream-auth-debug",
        )
        upstream_fixture = ServerFixture(upstream_base, upstream_stop, upstream_server)

        settings.set_base_settings(
            settings.Settings.model_validate(
                {
                    "dremio": {
                        "uri": "https://api.example.dremio.cloud",
                        "project_id": project_id,
                        "pat": "test-pat",
                        "auth_issuer_uri_override": upstream_base,
                    },
                    "tools": {"server_mode": ToolType.FOR_SELF.name},
                }
            )
        )

        mcp = init(
            transport=Transports.streamable_http,
            port=mcp_port,
            host="127.0.0.1",
            mode=settings.instance().tools.server_mode,
            support_project_id_endpoints=True,
            auth_debug=True,
        )
        mcp_app = mcp.streamable_http_app()
        mcp_server, mcp_stop = start_server_with_app(
            mcp_app,
            host="127.0.0.1",
            port=mcp_port,
            log_level="warning",
            name="mcp-auth-debug",
        )
        mcp_fixture = ServerFixture(
            f"http://127.0.0.1:{mcp_port}/mcp/{project_id}/", mcp_stop, mcp_server
        )

        yield mcp_fixture, upstream_state
    finally:
        if mcp_fixture is not None:
            mcp_fixture.close()
        if upstream_fixture is not None:
            upstream_fixture.close()
        settings.set_base_settings(old)


@pytest.mark.asyncio
async def test_auth_debug_rewrites_project_scoped_oauth_metadata(mock_config_dir):
    project_id = str(uuid.uuid4())
    async with auth_debug_server(project_id) as (mcp_fixture, _):
        parsed = urlparse(mcp_fixture.url)
        metadata_url = parsed._replace(
            path=f"/mcp/{project_id}/.well-known/oauth-authorization-server"
        ).geturl()

        async with AsyncClient(follow_redirects=False) as client:
            response = await client.get(metadata_url)

        assert response.status_code == 200, response.text
        data = response.json()
        origin = f"{parsed.scheme}://{parsed.netloc}"
        assert data["issuer"] == origin
        assert data["authorization_endpoint"] == f"{origin}/oauth/authorize"
        assert data["token_endpoint"] == f"{origin}/oauth/token"
        assert data["registration_endpoint"] == f"{origin}/oauth/register"


@pytest.mark.asyncio
async def test_auth_debug_proxies_token_and_register_requests(mock_config_dir):
    project_id = str(uuid.uuid4())
    async with auth_debug_server(project_id) as (mcp_fixture, upstream_state):
        parsed = urlparse(mcp_fixture.url)
        origin = f"{parsed.scheme}://{parsed.netloc}"

        async with AsyncClient(follow_redirects=False) as client:
            token_response = await client.post(
                f"{origin}/oauth/token",
                data={"grant_type": "refresh_token", "refresh_token": "rt-1"},
                headers={"x-client-header": "token-debug"},
            )
            register_response = await client.post(
                f"{origin}/oauth/register",
                json={
                    "client_name": "debug-client",
                    "redirect_uris": ["http://localhost/callback"],
                },
                headers={"x-client-header": "register-debug"},
            )

        assert token_response.status_code == 200, token_response.text
        assert token_response.json() == {
            "proxied": True,
            "grant_type": "refresh_token",
        }
        assert token_response.headers["x-upstream-auth-debug"] == "token"
        assert upstream_state.token_requests == [
            {
                "form": {
                    "grant_type": "refresh_token",
                    "refresh_token": "rt-1",
                },
                "headers": ANY,
            }
        ]
        assert (
            upstream_state.token_requests[0]["headers"]["x-client-header"]
            == "token-debug"
        )

        assert register_response.status_code == 200, register_response.text
        assert register_response.json() == {
            "client_id": "upstream-client",
            "client_name": "debug-client",
        }
        assert register_response.headers["x-upstream-auth-debug"] == "register"
        assert upstream_state.register_requests == [
            {
                "json": {
                    "client_name": "debug-client",
                    "redirect_uris": ["http://localhost/callback"],
                },
                "headers": ANY,
            }
        ]
        assert (
            upstream_state.register_requests[0]["headers"]["x-client-header"]
            == "register-debug"
        )


@pytest.mark.asyncio
async def test_auth_debug_normalizes_register_payload_for_claude(mock_config_dir):
    project_id = str(uuid.uuid4())
    async with auth_debug_server(project_id) as (mcp_fixture, upstream_state):
        parsed = urlparse(mcp_fixture.url)
        origin = f"{parsed.scheme}://{parsed.netloc}"

        async with AsyncClient(follow_redirects=False) as client:
            response = await client.post(
                f"{origin}/oauth/register",
                json={
                    "redirect_uris": ["https://claude.ai/api/mcp/auth_callback"],
                    "token_endpoint_auth_method": "none",
                    "grant_types": ["authorization_code", "refresh_token"],
                    "response_types": ["code"],
                    "scope": "dremio.all offline_access",
                    "client_name": "Claude",
                    "application_type": "web",
                },
            )

        assert response.status_code == 200, response.text
        assert upstream_state.register_requests == [
            {
                "json": {
                    "redirect_uris": ["https://claude.ai/api/mcp/auth_callback"],
                    "token_endpoint_auth_method": "none",
                    "grant_types": ["authorization_code", "refresh_token"],
                    "response_types": ["code"],
                    "scope": "dremio.all offline_access",
                    "client_name": "Claude",
                },
                "headers": ANY,
            }
        ]


@pytest.mark.asyncio
async def test_auth_debug_protected_resource_metadata_is_local(mock_config_dir):
    project_id = str(uuid.uuid4())
    async with auth_debug_server(project_id) as (mcp_fixture, _):
        metadata_url = _protected_resource_metadata_url(mcp_fixture.url)

        async with AsyncClient() as client:
            response = await client.get(metadata_url)

        assert response.status_code == 200, response.text
        data = response.json()
        parsed = urlparse(mcp_fixture.url)
        origin = f"{parsed.scheme}://{parsed.netloc}"
        assert data["resource"] == mcp_fixture.url.rstrip("/")
        assert data["authorization_servers"] == [origin]


@pytest.mark.asyncio
async def test_auth_debug_proxies_authorize_redirect(mock_config_dir):
    project_id = str(uuid.uuid4())
    async with auth_debug_server(project_id) as (mcp_fixture, upstream_state):
        parsed = urlparse(mcp_fixture.url)
        origin = f"{parsed.scheme}://{parsed.netloc}"
        redirect_uri = "http://localhost/callback"

        async with AsyncClient(follow_redirects=False) as client:
            response = await client.get(
                f"{origin}/oauth/authorize",
                params={
                    "client_id": "client-1",
                    "redirect_uri": redirect_uri,
                    "state": "debug-state",
                },
            )

        assert response.status_code == 302
        params = parse_qs(urlparse(response.headers["location"]).query)
        assert params["code"] == ["upstream-code"]
        assert params["state"] == ["debug-state"]
        assert upstream_state.authorize_requests == [
            {
                "client_id": "client-1",
                "redirect_uri": redirect_uri,
                "state": "debug-state",
            }
        ]


@pytest.mark.asyncio
async def test_auth_debug_metadata_is_local_when_upstream_is_unreachable(mock_config_dir):
    old = settings.instance()
    fixture = None
    try:
        project_id = str(uuid.uuid4())
        mcp_port = _reserve_local_port()
        settings.set_base_settings(
            settings.Settings.model_validate(
                {
                    "dremio": {
                        "uri": "https://api.example.dremio.cloud",
                        "project_id": project_id,
                        "pat": "test-pat",
                        "auth_issuer_uri_override": "https://definitely-invalid-host.invalid",
                    },
                    "tools": {"server_mode": ToolType.FOR_SELF.name},
                }
            )
        )
        mcp = init(
            transport=Transports.streamable_http,
            port=mcp_port,
            host="127.0.0.1",
            mode=settings.instance().tools.server_mode,
            support_project_id_endpoints=True,
            auth_debug=True,
        )
        app = mcp.streamable_http_app()
        server, stop_event = start_server_with_app(
            app,
            host="127.0.0.1",
            port=mcp_port,
            log_level="warning",
            name="mcp-auth-debug-unreachable",
        )
        fixture = ServerFixture(
            f"http://127.0.0.1:{mcp_port}/mcp/{project_id}/", stop_event, server
        )

        parsed = urlparse(fixture.url)
        metadata_url = parsed._replace(
            path=f"/mcp/{project_id}/.well-known/oauth-authorization-server"
        ).geturl()
        origin = f"{parsed.scheme}://{parsed.netloc}"
        async with AsyncClient(follow_redirects=False) as client:
            response = await client.get(metadata_url)

        assert response.status_code == 200, response.text
        assert response.json()["issuer"] == origin
        assert response.json()["authorization_endpoint"] == f"{origin}/oauth/authorize"
        assert response.json()["token_endpoint"] == f"{origin}/oauth/token"
        assert response.json()["registration_endpoint"] == f"{origin}/oauth/register"
    finally:
        if fixture is not None:
            fixture.close()
        settings.set_base_settings(old)
