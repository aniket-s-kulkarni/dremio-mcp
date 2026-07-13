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
from urllib.parse import urlparse

import pytest
from httpx import AsyncClient
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
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


@contextlib.asynccontextmanager
async def oauth_register_proxy_server() -> AsyncGenerator[tuple[ServerFixture, list[dict]], None]:
    old = settings.instance()
    upstream_fixture = None
    mcp_fixture = None
    try:
        register_requests: list[dict] = []
        upstream_port = _reserve_local_port()
        mcp_port = _reserve_local_port()
        while mcp_port == upstream_port:
            mcp_port = _reserve_local_port()

        upstream_base = f"http://127.0.0.1:{upstream_port}"

        async def register(request: Request):
            register_requests.append(
                {
                    "json": await request.json(),
                    "headers": dict(request.headers),
                }
            )
            return JSONResponse(
                {"client_id": "upstream-client", "client_name": "upstream-name"},
                headers={"x-upstream-register": "true"},
            )

        upstream_app = Starlette(
            routes=[Route("/oauth/register", register, methods=["POST"])]
        )
        upstream_server, upstream_stop = start_server_with_app(
            upstream_app,
            host="127.0.0.1",
            port=upstream_port,
            log_level="warning",
            name="upstream-oauth-register",
        )
        upstream_fixture = ServerFixture(upstream_base, upstream_stop, upstream_server)

        settings.set_base_settings(
            settings.Settings.model_validate(
                {
                    "dremio": {
                        "uri": "https://api.example.dremio.cloud",
                        "project_id": str(uuid.uuid4()),
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
        )
        mcp_server, mcp_stop = start_server_with_app(
            mcp.streamable_http_app(),
            host="127.0.0.1",
            port=mcp_port,
            log_level="warning",
            name="mcp-oauth-register",
        )
        mcp_fixture = ServerFixture(
            f"http://127.0.0.1:{mcp_port}/mcp/",
            mcp_stop,
            mcp_server,
        )

        yield mcp_fixture, register_requests
    finally:
        if mcp_fixture is not None:
            mcp_fixture.close()
        if upstream_fixture is not None:
            upstream_fixture.close()
        settings.set_base_settings(old)


@pytest.mark.asyncio
async def test_oauth_metadata_advertises_local_register_and_proxies_request(
    mock_config_dir,
):
    async with oauth_register_proxy_server() as (mcp_fixture, register_requests):
        parsed = urlparse(mcp_fixture.url)
        origin = f"{parsed.scheme}://{parsed.netloc}"
        metadata_url = parsed._replace(
            path="/.well-known/oauth-authorization-server"
        ).geturl()

        async with AsyncClient(follow_redirects=False) as client:
            metadata_response = await client.get(metadata_url)
            register_response = await client.post(
                f"{origin}/oauth/register",
                json={
                    "redirect_uris": ["https://claude.ai/api/mcp/auth_callback"],
                    "client_name": "Claude",
                    "scope": "dremio.all offline_access",
                    "grant_types": ["authorization_code", "refresh_token"],
                    "response_types": ["code"],
                    "application_type": "web",
                },
                headers={"x-client-header": "register-proxy"},
            )

        assert metadata_response.status_code == 200, metadata_response.text
        metadata = metadata_response.json()
        assert metadata["registration_endpoint"] == f"{origin}/oauth/register"

        assert register_response.status_code == 200, register_response.text
        assert register_response.json() == {
            "client_id": "upstream-client",
            "client_name": "upstream-name",
        }
        assert register_response.headers["x-upstream-register"] == "true"
        assert register_requests == [
            {
                "json": {
                    "redirect_uris": ["https://claude.ai/api/mcp/auth_callback"],
                    "client_name": "Claude",
                    "scope": "dremio.all offline_access",
                },
                "headers": ANY,
            }
        ]
        assert register_requests[0]["headers"]["x-client-header"] == "register-proxy"
