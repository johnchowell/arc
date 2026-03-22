from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from semantic_searcher.config import settings

engine = create_async_engine(
    settings.database_url,
    pool_size=30,
    max_overflow=20,
    pool_recycle=1800,
    pool_pre_ping=True,
)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def get_session() -> AsyncSession:
    async with async_session() as session:
        yield session
