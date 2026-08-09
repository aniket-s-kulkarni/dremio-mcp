#
#  Copyright (C) 2017-2026 Dremio Corporation
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

"""Minimal local stdio MCP server for manual Claude/Desktop testing."""

from __future__ import annotations

import logging
import os
import sys
from inspect import Parameter, Signature
from typing import Any
from urllib.parse import quote

import requests
from mcp.server.fastmcp import FastMCP

from dremioai.config import settings
from dremioai.config.tools import ToolType
from dremioai.tools.tools import (
    GetDescriptionOfTableOrSchema,
    GetSchemaOfTable,
    RunSqlQuery,
)


LOG_PATH = "/tmp/dremio-local.log"
LOGGER_NAME = "dremio_local_stdio_mcp"
REQUIRED_ENV_VARS = ("DREMIO_PROJECT_ID", "DREMIO_TOKEN", "DREMIO_URL")


def configure_logging() -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(LOG_PATH)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    logger.addHandler(handler)
    logger.propagate = False
    return logger


logger = configure_logging()


def redact_token(token: str) -> str:
    if len(token) <= 8:
        return "*" * len(token)
    return f"{token[:4]}...{token[-4:]}"


def require_env() -> dict[str, str]:
    values: dict[str, str] = {}
    missing = [name for name in REQUIRED_ENV_VARS if not os.getenv(name)]
    if missing:
        message = f"Missing required environment variables: {', '.join(missing)}"
        logger.error(message)
        raise RuntimeError(message)

    for name in REQUIRED_ENV_VARS:
        values[name] = os.environ[name]

    log_event(
        "environment_loaded",
        project_id=values["DREMIO_PROJECT_ID"],
        url=values["DREMIO_URL"],
        token=redact_token(values["DREMIO_TOKEN"]),
    )
    return values


def log_event(event: str, **fields: Any) -> None:
    details = " ".join(f"{key}={value!r}" for key, value in sorted(fields.items()))
    logger.info("%s %s", event, details)


def configure_dremioai_settings() -> settings.Settings:
    inst = settings.Settings.model_validate(
        {
            "log_level": "INFO",
            "dremio": {
                "uri": ENV["DREMIO_URL"],
                "pat": ENV["DREMIO_TOKEN"],
                "project_id": ENV["DREMIO_PROJECT_ID"],
                "enable_search": True,
            },
            "tools": {"server_mode": ToolType.FOR_DATA_PATTERNS.name},
        }
    )
    settings.set_base_settings(inst, initialize_ld=False)
    log_event(
        "dremioai_settings_configured",
        uri=ENV["DREMIO_URL"],
        project_id=ENV["DREMIO_PROJECT_ID"],
        server_mode=ToolType.FOR_DATA_PATTERNS.name,
    )
    return inst


def auth_headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {ENV['DREMIO_TOKEN']}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }


def dremio_endpoint(path: str) -> str:
    return f"{ENV['DREMIO_URL'].rstrip('/')}{path}"


def map_schema_type(schema: dict[str, Any] | None) -> Any:
    schema = schema or {}
    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        non_null = [item for item in schema_type if item != "null"]
        schema_type = non_null[0] if non_null else None

    return {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "array": list,
        "object": dict,
    }.get(schema_type, Any)


def build_signature(input_schema: dict[str, Any]) -> Signature:
    if input_schema.get("type") not in (None, "object"):
        raise RuntimeError(f"Unsupported tool schema type: {input_schema.get('type')!r}")

    properties = input_schema.get("properties", {})
    required = set(input_schema.get("required", []))
    parameters: list[Parameter] = []

    for name, schema in properties.items():
        annotation = map_schema_type(schema)
        default = Parameter.empty if name in required else schema.get("default", None)
        parameters.append(
            Parameter(
                name=name,
                kind=Parameter.KEYWORD_ONLY,
                default=default,
                annotation=annotation,
            )
        )

    return Signature(parameters=parameters)


def fetch_ai_tool(tool_name: str) -> dict[str, Any]:
    url = dremio_endpoint(f"/v1/projects/{ENV['DREMIO_PROJECT_ID']}/ai/tools")
    log_event("fetch_ai_tools_started", url=url, tool_name=tool_name)
    response = requests.get(url, headers=auth_headers(), timeout=30)
    response.raise_for_status()
    payload = response.json()
    tools = payload.get("tools", [])
    log_event("fetch_ai_tools_succeeded", tool_count=len(tools), tool_name=tool_name)

    for tool in tools:
        if tool.get("name") == tool_name:
            log_event(
                "tool_shape_discovered",
                tool_name=tool_name,
                description=tool.get("description", ""),
                input_schema=tool.get("inputSchema", {}),
            )
            return tool

    raise RuntimeError(f"Tool '{tool_name}' not found in Dremio AI tools list")


def build_remote_tool(tool_def: dict[str, Any]):
    tool_name = tool_def["name"]
    input_schema = tool_def.get("inputSchema") or {"type": "object"}
    description = tool_def.get("description") or f"Proxy to Dremio AI tool '{tool_name}'."
    signature = build_signature(input_schema)

    async def remote_tool(**kwargs: Any) -> dict[str, Any]:
        invoke_url = dremio_endpoint(
            f"/v1/projects/{ENV['DREMIO_PROJECT_ID']}/ai/tools/{quote(tool_name, safe='')}:invoke"
        )
        payload = {"args": kwargs}
        log_event(
            "tool_called",
            tool=tool_name,
            args=kwargs,
            invoke_url=invoke_url,
        )
        response = requests.post(
            invoke_url,
            headers=auth_headers(),
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
        result = response.json()
        log_event("tool_succeeded", tool=tool_name, result=result)
        return result

    remote_tool.__name__ = tool_name
    remote_tool.__doc__ = description
    remote_tool.__signature__ = signature
    return remote_tool, description


def register_local_dremioai_tool(
    tool_instance: Any,
    *,
    name: str,
    description: str | None = None,
) -> None:
    SERVER.add_tool(
        tool_instance.invoke,
        name=name,
        description=description or (tool_instance.invoke.__doc__ or ""),
    )
    log_event("local_tool_registered", tool=name)

ENV = require_env()
SETTINGS = configure_dremioai_settings()
SERVER = FastMCP(
    "Dremio Local Test",
    log_level="ERROR",
    debug=True,
)

SEARCH_VIEWS_AND_TABLES = fetch_ai_tool("searchViewsAndTables")


@SERVER.tool(name="hello", description="Say hello from the local Dremio MCP test server.")
async def hello(name: str = "world") -> dict[str, str]:
    log_event("tool_called", tool="hello", name=name)
    result = {
        "message": f"hello, {name}",
        "project_id": ENV["DREMIO_PROJECT_ID"],
        "url": ENV["DREMIO_URL"],
    }
    log_event("tool_succeeded", tool="hello", result=result)
    return result


search_views_and_tables, search_views_and_tables_description = build_remote_tool(
    SEARCH_VIEWS_AND_TABLES
)
SERVER.add_tool(
    search_views_and_tables,
    name="searchViewsAndTables",
    description=search_views_and_tables_description,
)

register_local_dremioai_tool(
    GetDescriptionOfTableOrSchema(),
    name="GetDescriptionOfTableOrSchema",
)
register_local_dremioai_tool(
    GetSchemaOfTable(),
    name="GetSchemaOfTable",
)
register_local_dremioai_tool(
    RunSqlQuery(),
    name="RunSql",
    description=RunSqlQuery.invoke.__doc__ or "",
)


def _handle_excepthook(exc_type, exc_value, exc_traceback) -> None:
    logger.exception(
        "Unhandled exception",
        exc_info=(exc_type, exc_value, exc_traceback),
    )


sys.excepthook = _handle_excepthook


def main() -> None:
    log_event(
        "server_initialized",
        project_id=ENV["DREMIO_PROJECT_ID"],
        url=ENV["DREMIO_URL"],
        token=redact_token(ENV["DREMIO_TOKEN"]),
        log_path=LOG_PATH,
        tools=[
            "hello",
            "searchViewsAndTables",
            "GetDescriptionOfTableOrSchema",
            "GetSchemaOfTable",
            "RunSql",
        ],
    )
    SERVER.run(transport="stdio")


if __name__ == "__main__":
    main()
