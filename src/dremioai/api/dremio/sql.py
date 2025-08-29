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
from contextlib import asynccontextmanager
from urllib.parse import urlparse, ParseResult

from pydantic import BaseModel, Field
from typing import List, Dict, Union, Optional, Any, AsyncGenerator, Tuple

from enum import auto
from datetime import datetime
from dremioai.api.util import UStrEnum, run_in_parallel

import pandas as pd
import asyncio
import itertools
import warnings

from dremioai.api.transport import DremioAsyncHttpClient as AsyncHttpClient
from dremioai.config import settings
from adbc_driver_flightsql import dbapi
from adbc_driver_flightsql import DatabaseOptions, ConnectionOptions


class ArcticSourceType(UStrEnum):
    BRANCH = auto()
    TAG = auto()
    COMMIT = auto()


class ArcticSource(BaseModel):
    type: ArcticSourceType = Field(..., alias="type")
    value: str


class Query(BaseModel):
    sql: str = Field(..., alias="sql")
    context: Optional[List[str]] = None
    references: Optional[Dict[str, ArcticSource]] = None


class QuerySubmission(BaseModel):
    id: str


class JobState(UStrEnum):
    NOT_SUBMITTED = auto()
    STARTING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    CANCELED = auto()
    FAILED = auto()
    CANCELLATION_REQUESTED = auto()
    PLANNING = auto()
    PENDING = auto()
    METADATA_RETRIEVAL = auto()
    QUEUED = auto()
    ENGINE_START = auto()
    EXECUTION_PLANNING = auto()
    INVALID_STATE = auto()


class QueryType(UStrEnum):
    UI_RUN = auto()
    UI_PREVIEW = auto()
    UI_INTERNAL_PREVIEW = auto()
    UI_INTERNAL_RUN = auto()
    UI_EXPORT = auto()
    ODBC = auto()
    JDBC = auto()
    REST = auto()
    ACCELERATOR_CREATE = auto()
    ACCELERATOR_DROP = auto()
    UNKNOWN = auto()
    PREPARE_INTERNAL = auto()
    ACCELERATOR_EXPLAIN = auto()
    UI_INITIAL_PREVIEW = auto()


class Relationship(UStrEnum):
    CONSIDERED = auto()
    MATCHED = auto()
    CHOSEN = auto()


class ReflectionReleationShips(BaseModel):
    dataset_id: str = Field(..., alias="datasetId")
    reflection_id: str = Field(..., alias="reflectionId")
    relationship: Relationship


class Acceleration(BaseModel):
    reflection_relationships: List[ReflectionReleationShips] = Field(
        ..., alias="reflectionRelationships"
    )


class Job(BaseModel):
    job_state: JobState = Field(..., alias="jobState")
    row_count: Optional[int] = Field(default=0, alias="rowCount")
    error_message: Optional[str] = Field(default=None, alias="errorMessage")
    started_at: Optional[datetime] = Field(default=None, alias="startedAt")
    ended_at: Optional[datetime] = Field(default=None, alias="endedAt")
    acceleration: Optional[Acceleration] = None
    query_type: QueryType = Field(..., alias="queryType")
    queue_name: Optional[str] = Field(default=None, alias="queueName")
    queue_id: Optional[str] = Field(default=None, alias="queueId")
    resource_scheduling_started_at: Optional[datetime] = Field(
        default=None, alias="resourceSchedulingStartedAt"
    )
    resource_scheduling_ended_at: Optional[datetime] = Field(
        default=None, alias="resourceSchedulingEndedAt"
    )
    cancellation_reason: Optional[str] = Field(default=None, alias="cancellationReason")

    @property
    def done(self):
        return self.job_state in {
            JobState.COMPLETED,
            JobState.CANCELED,
            JobState.FAILED,
        }

    @property
    def succeeded(self):
        return self.job_state == JobState.COMPLETED


class ResultSchemaType(BaseModel):
    name: str


class ResultSchema(BaseModel):
    name: str
    type: ResultSchemaType


class JobResults(BaseModel):
    row_count: int = Field(..., alias="rowCount")
    result_schema: Optional[List[ResultSchema]] = Field(..., alias="schema")
    rows: List[Dict[str, Any]]


class JobResultsWrapper(List[JobResults]):
    pass


class JobResultsParams(BaseModel):
    offset: Optional[int] = 0
    limit: Optional[int] = 500


async def _fetch_results(
    uri: str, pat: str, project_id: str, job_id: str, off: int, limit: int
) -> JobResults:
    client = AsyncHttpClient()
    params = JobResultsParams(offset=off, limit=limit)
    endpoint = f"/v0/projects/{project_id}" if project_id else "/api/v3"
    return await client.get(
        f"{endpoint}/job/{job_id}/results",
        params=params.model_dump(),
        deser=JobResults,
    )


async def get_results(
    project_id: str,
    qs: Union[QuerySubmission, str],
    use_df: bool = False,
    uri: Optional[str] = None,
    pat: Optional[str] = None,
    client: Optional[AsyncHttpClient] = None,
) -> JobResultsWrapper:
    if isinstance(qs, str):
        qs = QuerySubmission(id=qs)

    if client is None:
        client = AsyncHttpClient()

    endpoint = f"/v0/projects/{project_id}" if project_id else "/api/v3"
    job: Job = await client.get(f"{endpoint}/job/{qs.id}", deser=Job)
    while not job.done:
        await asyncio.sleep(0.5)
        job = await client.get(f"{endpoint}/job/{qs.id}", deser=Job)

    if not job.succeeded:
        emsg = (
            job.error_message
            if job.error_message
            else (
                job.cancellation_reason
                if job.job_state == JobState.CANCELED
                else "Unknown error"
            )
        )
        raise RuntimeError(f"Job {qs.id} failed: {emsg}")

    if job.row_count == 0:
        return pd.DataFrame() if use_df else JobResultsWrapper([])

    limit = min(500, job.row_count)

    results = await run_in_parallel(
        [
            _fetch_results(uri, pat, project_id, qs.id, off, limit)
            for off in range(0, job.row_count, limit)
        ]
    )
    jr = JobResultsWrapper(itertools.chain(r for r in results))

    if use_df:
        df = pd.DataFrame(
            data=itertools.chain.from_iterable(jr.rows for jr in jr),
            columns=[rs.name for rs in jr[0].result_schema],
        )
        for rs in jr[0].result_schema:
            if rs.type.name == "TIMESTAMP":
                df[rs.name] = pd.to_datetime(df[rs.name])
        return df

    return jr


def convert_to_adbc_uri(uri: str, is_dc: bool = False) -> ParseResult:
    u = urlparse(uri)
    if is_dc and u.netloc.startswith("api."):
        u = u._replace(netloc=f"data.{u.netloc[4:]}")
    return u._replace(scheme="grpc+tls")


def create_adbc_connection_options(
    pat: str, project_id: str = None
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    session_options = {"routing_tag": "MCP"}
    db_args = {
        DatabaseOptions.AUTHORIZATION_HEADER.value: f"Bearer {pat}",
        f"{DatabaseOptions.RPC_CALL_HEADER_PREFIX.value}useEncryption": "true",
    }
    if project_id:
        db_args[f"{DatabaseOptions.RPC_CALL_HEADER_PREFIX.value}Cookie"] = (
            f"project_id={project_id}"
        )
    return db_args, {
        f"{ConnectionOptions.OPTION_SESSION_OPTION_PREFIX.value}{key}": value
        for key, value in session_options.items()
    }


@asynccontextmanager
async def adbc_connect(
    uri: str, db_args: Dict[str, Any], conn_args: Dict[str, Any]
) -> AsyncGenerator[dbapi.Connection, None]:
    conn = None
    try:
        # Suppress the autocommit warning from ADBC driver manager
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Cannot disable autocommit; conn will not be DB-API 2.0 compliant",
                category=UserWarning,
            )
            conn = await asyncio.to_thread(
                dbapi.connect, uri=uri, db_kwargs=db_args, conn_kwargs=conn_args
            )
        yield conn
    finally:
        if conn is not None:
            await asyncio.to_thread(conn.close)


@asynccontextmanager
async def adbc_cursor(conn: dbapi.Connection) -> AsyncGenerator[dbapi.Cursor, None]:
    cursor = None
    try:
        cursor = await asyncio.to_thread(conn.cursor)
        yield cursor
    finally:
        if cursor is not None:
            await asyncio.to_thread(cursor.close)


def convert_to_job_results(schema: List[Tuple], rows: List[Tuple]) -> JobResultsWrapper:
    rs = [
        ResultSchema(name=info[0], type=ResultSchemaType(name=str(info[1])))
        for info in schema
    ]

    def convert_row(row: Tuple) -> Dict[str, Any]:
        return {rs[ix].name: col for ix, col in enumerate(row)}

    rsrows = [convert_row(row) for row in rows]
    return JobResultsWrapper([JobResults(rowCount=len(rows), schema=rs, rows=rsrows)])


async def run_adbc_query(
    query: Union[Query, str], use_df: bool = False
) -> Union[JobResultsWrapper, pd.DataFrame]:
    if not isinstance(query, Query):
        query = Query(sql=query)

    uri = convert_to_adbc_uri(
        settings.instance().dremio.uri, is_dc=settings.instance().dremio.is_cloud
    ).geturl()
    db_args, conn_args = create_adbc_connection_options(
        settings.instance().dremio.pat, settings.instance().dremio.project_id
    )

    async with adbc_connect(uri, db_args, conn_args) as conn:
        async with adbc_cursor(conn) as cursor:
            await asyncio.to_thread(cursor.execute, query.sql)
            if use_df:
                return await asyncio.to_thread(cursor.fetch_df)
            rows = await asyncio.to_thread(cursor.fetchall)
            return convert_to_job_results(cursor.description, rows)


async def run_query(
    query: Union[Query, str], use_df: bool = False, use_adbc: bool = False
) -> Union[JobResultsWrapper, pd.DataFrame]:
    if use_adbc:
        return await run_adbc_query(query, use_df=use_df)

    client = AsyncHttpClient()
    if not isinstance(query, Query):
        query = Query(sql=query)

    project_id = settings.instance().dremio.project_id
    endpoint = f"/v0/projects/{project_id}" if project_id else "/api/v3"
    qs: QuerySubmission = await client.post(
        f"{endpoint}/sql", body=query.model_dump(), deser=QuerySubmission
    )
    return await get_results(project_id, qs, use_df=use_df, client=client)
