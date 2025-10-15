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

import pytest
from unittest.mock import patch, MagicMock

from dremioai.config.tools import ToolType
from dremioai.servers import mcp as mcp_server
from dremioai.tools.tools import get_tools
from dremioai.config import settings

from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.shared.auth import OAuthMetadata
from contextlib import asynccontextmanager, contextmanager
from rich import print as pp
from tempfile import TemporaryDirectory
from pathlib import Path
import json
from starlette.testclient import TestClient


@contextmanager
def mock_settings(mode: ToolType):
    """Create mock settings for testing MCP server"""
    # Create a mock settings instance
    try:
        old = settings.instance()
        with TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            settings._settings.set(
                settings.Settings.model_validate(
                    {
                        "dremio": {
                            "uri": "https://test-dremio-uri.com",
                            "pat": "test-pat",
                        },
                        "tools": {"server_mode": mode},
                    }
                )
            )
            cfg = temp_dir / "config.yaml"
            settings.write_settings(cfg=cfg, inst=settings.instance())
            yield settings.instance(), cfg
    finally:
        settings._settings.set(old)


@asynccontextmanager
async def mcp_server_session(cfg: Path):
    """Create an MCP server instance with mock settings"""
    params = mcp_server.create_default_mcpserver_config()
    params["args"].extend(["--cfg", str(cfg)])
    params = StdioServerParameters(command=params["command"], args=params["args"])
    async with (
        stdio_client(params) as (read, write),
        ClientSession(read, write) as session,
    ):
        await session.initialize()
        yield session


@pytest.mark.parametrize(
    "mode",
    [
        pytest.param(ToolType.FOR_SELF, id="FOR_SELF"),
        pytest.param(ToolType.FOR_DATA_PATTERNS, id="FOR_DATA_PATTERNS"),
        pytest.param(
            ToolType.FOR_SELF | ToolType.FOR_DATA_PATTERNS,
            id="FOR_SELF|FOR_DATA_PATTERNS",
        ),
    ],
)
@pytest.mark.asyncio
async def test_mcp_server_initialization(mode: ToolType):
    with mock_settings(mode) as (_, cfg):
        async with mcp_server_session(cfg) as session:
            tools = await session.list_tools()
            assert len(tools.tools) > 0
            names = {tool.name for tool in tools.tools}
            exp = {t.__name__ for t in get_tools(For=mode)}
            assert names == exp


@pytest.fixture(
    params=[pytest.param(True, id="exists"), pytest.param(False, id="not_exists")]
)
def claude_config_path(request):
    with TemporaryDirectory() as temp_dir:
        p = Path(temp_dir) / "claude_desktop_config.json"
        if request.param:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("{}")
        with patch("dremioai.servers.mcp.get_claude_config_path") as mock_update:
            mock_update.return_value = p
            yield p


def test_claude_config_creation(claude_config_path):
    dcmp = {"Dremio": mcp_server.create_default_mcpserver_config()}
    mcp_server.create_default_config_helper(False)

    assert claude_config_path.exists()
    d = json.load(claude_config_path.open())
    assert d["mcpServers"] == dcmp


@pytest.mark.parametrize(
    "endpoint",
    [
        pytest.param("/.well-known/oauth-authorization-server", id="oauth-server"),
        pytest.param("/.well-known/openid-configuration", id="openid-config"),
    ],
)
def test_well_known_endpoints(endpoint):
    """Test that both OAuth and OpenID well-known endpoints return the same metadata"""
    with mock_settings(ToolType.FOR_SELF) as (mock_inst, _):
        # Mock the auth configuration
        mock_inst.dremio.auth_issuer_uri_override = "https://test-issuer.com"

        # Create the MCP server with streamable_http transport
        server = mcp_server.init(
            mode=[ToolType.FOR_SELF],
            transport=mcp_server.Transports.streamable_http,
            port=8000,
        )

        # Get the app from the server
        app = server.streamable_http_app()

        # Create a test client
        client = TestClient(app)

        # Test the endpoint
        response = client.get(endpoint)

        # Should return 200 with valid OAuth metadata
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"

        # Validate the response structure
        metadata = OAuthMetadata.model_validate(response.json())
        assert str(metadata.issuer).rstrip('/') == "https://test-issuer.com"
        assert metadata.scopes_supported == ["dremio.all", "offline_access"]
        assert metadata.response_types_supported == ["code"]
        assert metadata.grant_types_supported == ["authorization_code", "refresh_token"]
        assert metadata.code_challenge_methods_supported == ["S256"]
        assert metadata.token_endpoint_auth_methods_supported == ["client_secret_post"]


def test_well_known_endpoints_without_auth():
    """Test that well-known endpoints return 404 when auth is not configured"""
    with mock_settings(ToolType.FOR_SELF) as (mock_inst, _):
        # Ensure auth is not configured
        mock_inst.dremio.auth_issuer_uri_override = None

        # Create the MCP server with streamable_http transport
        server = mcp_server.init(
            mode=[ToolType.FOR_SELF],
            transport=mcp_server.Transports.streamable_http,
            port=8000,
        )

        # Get the app from the server
        app = server.streamable_http_app()

        # Create a test client
        client = TestClient(app)

        # Test both endpoints
        for endpoint in ["/.well-known/oauth-authorization-server", "/.well-known/openid-configuration"]:
            response = client.get(endpoint)
            assert response.status_code == 404


def test_oauth_registration_endpoint():
    """Test the OAuth client registration endpoint following RFC 7591"""
    with mock_settings(ToolType.FOR_SELF) as (mock_inst, _):
        # Mock the auth configuration
        mock_inst.dremio.auth_issuer_uri_override = "https://test-issuer.com"

        # Create the MCP server with streamable_http transport
        server = mcp_server.init(
            mode=[ToolType.FOR_SELF],
            transport=mcp_server.Transports.streamable_http,
            port=8000,
        )

        # Get the app from the server
        app = server.streamable_http_app()

        # Create a test client
        client = TestClient(app)

        # First, verify the registration endpoint is in the metadata
        metadata_response = client.get("/.well-known/oauth-authorization-server")
        assert metadata_response.status_code == 200
        metadata = metadata_response.json()
        assert "registration_endpoint" in metadata
        assert metadata["registration_endpoint"].endswith("/oauth/register")

        # Test the registration endpoint with RFC 7591 compliant data
        registration_data = {
            "client_name": "Test Client",
            "redirect_uris": ["http://localhost:8976"],
            "grant_types": ["authorization_code", "refresh_token"],
            "response_types": ["code"],
            "scope": "dremio.all offline_access",
            "token_endpoint_auth_method": "client_secret_post"
        }

        response = client.post(
            "/oauth/register",
            json=registration_data
        )

        # Should return 201 Created
        assert response.status_code == 201
        assert response.headers["content-type"] == "application/json"

        # Validate the response follows RFC 7591 (OAuthClientInformationFull)
        response_json = response.json()

        # Check required fields from OAuthClientInformationFull
        assert response_json["client_id"] == "172bdf86-28ca-4933-b46f-9519b413271c"
        assert "client_id_issued_at" in response_json
        assert "client_secret_expires_at" in response_json

        # Check that the client metadata was preserved in the response
        assert response_json["client_name"] == "Test Client"
        # Note: Pydantic's AnyUrl normalizes URLs by adding trailing slashes
        assert response_json["redirect_uris"] == ["http://localhost:8976/"]
        assert response_json["grant_types"] == ["authorization_code", "refresh_token"]
        assert response_json["response_types"] == ["code"]
        assert response_json["scope"] == "dremio.all offline_access"
        assert response_json["token_endpoint_auth_method"] == "client_secret_post"


def test_oauth_registration_endpoint_validation():
    """Test that the registration endpoint validates input properly"""
    with mock_settings(ToolType.FOR_SELF) as (mock_inst, _):
        # Mock the auth configuration
        mock_inst.dremio.auth_issuer_uri_override = "https://test-issuer.com"

        # Create the MCP server with streamable_http transport
        server = mcp_server.init(
            mode=[ToolType.FOR_SELF],
            transport=mcp_server.Transports.streamable_http,
            port=8000,
        )

        # Get the app from the server
        app = server.streamable_http_app()

        # Create a test client
        client = TestClient(app)

        # Test with invalid JSON
        response = client.post(
            "/oauth/register",
            content="invalid json{",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 400
        assert "error" in response.json()
        assert response.json()["error"] == "invalid_request"

        # Test with missing required field (redirect_uris)
        response = client.post(
            "/oauth/register",
            json={
                "client_name": "Test Client",
                # Missing redirect_uris
            }
        )
        assert response.status_code == 400
        assert "error" in response.json()
        assert response.json()["error"] == "invalid_client_metadata"

        # Test with invalid grant_type
        response = client.post(
            "/oauth/register",
            json={
                "redirect_uris": ["http://localhost:8976"],
                "grant_types": ["implicit"],  # Not supported
            }
        )
        assert response.status_code == 400
        assert "error" in response.json()
        assert response.json()["error"] == "invalid_client_metadata"
