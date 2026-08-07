"""
NeonDB (Postgres) connection layer.

- `engine` / `AsyncSessionLocal` -> normal app CRUD via SQLAlchemy (uses pooled URL).
- `get_sync_conninfo()` -> plain conninfo string for the LangGraph Postgres
  checkpointer, which needs a direct (unpooled) connection.
"""
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

from app.config import settings

from urllib.parse import urlsplit, urlunsplit, parse_qs, urlencode


# Handle Neon/PG URLs that include `sslmode` or `channel_binding` query params.
# asyncpg.connect doesn't accept `sslmode` directly; translate it into the
# `ssl` connect arg and remove unsupported query params from the URL.
def _prepare_async_url(url: str):
    parts = urlsplit(url)
    qs = parse_qs(parts.query)
    sslmode = qs.pop("sslmode", None)
    # remove channel_binding too if present
    qs.pop("channel_binding", None)

    new_query = urlencode({k: v[0] for k, v in qs.items()})
    clean = urlunsplit((parts.scheme, parts.netloc, parts.path, new_query, parts.fragment))
    async_url = clean.replace("postgresql://", "postgresql+asyncpg://", 1)
    connect_args = {}
    if sslmode:
        # any value for sslmode -> enable SSL
        connect_args["ssl"] = True
    return async_url, connect_args


_async_url, _connect_args = _prepare_async_url(settings.database_url)

engine = create_async_engine(_async_url, pool_pre_ping=True, pool_size=5, max_overflow=10, connect_args=_connect_args)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)


async def get_session() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        yield session


def get_checkpointer_conninfo() -> str:
    """Direct/unpooled connection string, required by langgraph-checkpoint-postgres."""
    return settings.database_url_unpooled


async def init_schema():
    """Run schema.sql once at startup (idempotent, uses IF NOT EXISTS)."""
    import pathlib
    schema_path = pathlib.Path(__file__).parent / "schema.sql"
    sql = schema_path.read_text()
    async with engine.begin() as conn:
        for statement in sql.split(";"):
            statement = statement.strip()
            if statement:
                await conn.exec_driver_sql(statement)
