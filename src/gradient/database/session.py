# src/gradient/database/session.py

import datetime  # <--- Add datetime
import logging
from collections.abc import AsyncGenerator  # <--- Add Optional
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from gradient.core.config import get_data_dir

# Import Base and models used by the helper function
from .models import Base, ExperimentRun  # <--- Import ExperimentRun

logger = logging.getLogger(__name__)

# --- Database Configuration ---
DEFAULT_DB_SUBDIR = "database"
DEFAULT_DB_FILENAME = "gradient_data.db"


def get_database_url() -> str:
    db_dir = get_data_dir() / DEFAULT_DB_SUBDIR
    db_dir.mkdir(parents=True, exist_ok=True)
    db_path = db_dir / DEFAULT_DB_FILENAME
    db_url = f"sqlite+aiosqlite:///{db_path.resolve()}"
    logger.info(f"Using database URL: {db_url}")
    return db_url


DATABASE_URL = get_database_url()

# --- SQLAlchemy Async Engine ---
try:
    async_engine = create_async_engine(
        DATABASE_URL, echo=False, future=True, connect_args={"check_same_thread": False}
    )
except Exception as e:
    logger.critical(f"Failed to create async database engine: {e}", exc_info=True)
    async_engine = None

if async_engine is not None:
    from sqlalchemy import event

    @event.listens_for(async_engine.sync_engine, "connect")
    def _sqlite_pragmas(dbapi_connection, _record):
        # WAL: the gallery can read while the render queue writes; wait instead of failing
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA busy_timeout=15000")
        cursor.close()


async def dispose_engine() -> None:
    """Close pooled connections (at shutdown; also keeps tests' event loops apart)."""
    if async_engine is not None:
        await async_engine.dispose()


# --- Async Session Factory ---
AsyncSessionFactory = (
    async_sessionmaker(
        bind=async_engine,
        autocommit=False,
        autoflush=False,
        expire_on_commit=False,
        class_=AsyncSession,
    )
    if async_engine
    else None
)


async def init_db():
    if not async_engine:
        logger.error("Database engine not initialized. Cannot initialize DB.")
        return
    logger.info("Initializing database and creating tables...")
    async with async_engine.begin() as conn:
        try:
            await conn.run_sync(Base.metadata.create_all)
            added = await conn.run_sync(_add_missing_columns)
            if added:
                logger.info("Database upgraded: added columns %s", ", ".join(added))
            await conn.run_sync(_backfill_outputs)
            logger.info("Database tables created successfully (if they didn't exist).")
        except Exception as e:
            logger.error(f"Error creating database tables: {e}", exc_info=True)


def _add_missing_columns(sync_conn) -> list[str]:
    """create_all() never alters existing tables: add columns that newer versions define."""
    from sqlalchemy import inspect, text

    inspector = inspect(sync_conn)
    added: list[str] = []
    for table in Base.metadata.sorted_tables:
        if not inspector.has_table(table.name):
            continue
        existing = {c["name"] for c in inspector.get_columns(table.name)}
        for col in table.columns:
            if col.name in existing:
                continue
            ddl = f'ALTER TABLE "{table.name}" ADD COLUMN "{col.name}" '
            ddl += col.type.compile(dialect=sync_conn.dialect)
            if col.server_default is not None:
                ddl += f" NOT NULL DEFAULT {col.server_default.arg}"
            sync_conn.execute(text(ddl))
            if col.index:
                sync_conn.execute(
                    text(
                        f'CREATE INDEX IF NOT EXISTS "ix_{table.name}_{col.name}" '
                        f'ON "{table.name}" ("{col.name}")'
                    )
                )
            added.append(f"{table.name}.{col.name}")
    return added


def _backfill_outputs(sync_conn) -> None:
    """Fill the gallery columns of outputs logged before v0.5 and move legacy comma tags."""
    import json

    from sqlalchemy import text

    from gradient.gallery.metadata import output_fields

    rows = sync_conn.execute(
        text(
            "SELECT id, parameters, user_tags FROM generated_outputs "
            "WHERE model_id IS NULL AND parameters IS NOT NULL"
        )
    ).fetchall()
    for out_id, params, user_tags in rows:
        try:
            p = json.loads(params) if isinstance(params, str) else (params or {})
        except ValueError:
            p = {}
        fields = output_fields(p)
        fields["model_id"] = fields["model_id"] or ""
        sync_conn.execute(
            text(
                "UPDATE generated_outputs SET model_id=:model_id, prompt=:prompt, seed=:seed, "
                "width=:width, height=:height WHERE id=:id"
            ),
            {**fields, "id": out_id},
        )
        for tag in {t.strip() for t in (user_tags or "").split(",") if t.strip()}:
            sync_conn.execute(
                text(
                    "INSERT OR IGNORE INTO output_tags (output_id, tag, source, added) "
                    "VALUES (:id, :tag, 'manual', CURRENT_TIMESTAMP)"
                ),
                {"id": out_id, "tag": tag},
            )
    if rows:
        logger.info("Backfilled gallery fields for %d earlier outputs", len(rows))


@asynccontextmanager
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    if AsyncSessionFactory is None:
        logger.error("AsyncSessionFactory not initialized. Cannot get DB session.")
        raise RuntimeError("Database session factory not available.")
    session: AsyncSession = AsyncSessionFactory()
    logger.debug("DB Session created.")
    try:
        yield session
        await session.commit()
        logger.debug("DB Session committed.")
    except Exception as e:
        logger.error(f"Error during DB session: {e}. Rolling back.", exc_info=True)
        await session.rollback()
        raise
    except BaseException:
        # cancelled (e.g. at shutdown): roll back so no write lock is left behind
        await session.rollback()
        raise
    finally:
        await session.close()
        logger.debug("DB Session closed.")


async def db_session_dependency() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency: `db: AsyncSession = Depends(db_session_dependency)`."""
    async with get_db_session() as session:
        yield session


# --- Run status helper --- vvv
async def update_run_status(
    run_db_id: int, status: str, error: str | None = None, end_time: float | None = None
):
    """Updates the status and end time of an ExperimentRun record."""
    # Note: This function implicitly uses the get_db_session context manager
    try:
        async with get_db_session() as db:
            # Use db.get which is preferred for fetching by primary key
            run = await db.get(ExperimentRun, run_db_id)
            if run:
                run.status = status
                if error:
                    # Append error to notes? Or have a dedicated error field?
                    # For now, appending to notes might overwrite existing notes.
                    # Let's just set it directly for simplicity, overwriting old notes.
                    run.user_notes = f"Error: {error}"
                else:
                    # Clear notes if status is now completed without error? Optional.
                    # run.user_notes = None
                    pass
                if end_time:
                    run.end_time = datetime.datetime.fromtimestamp(end_time, tz=datetime.UTC)

                # No need to db.add(run) when modifying an existing tracked object
                logger.info(f"Updated ExperimentRun ID {run_db_id} status to '{status}'")
            else:
                logger.warning(f"Could not find ExperimentRun ID {run_db_id} to update status.")
    except Exception as e:
        # Log error but don't rollback here, rollback handled by get_db_session
        logger.error(f"Failed to update DB status for run ID {run_db_id}: {e}", exc_info=True)
