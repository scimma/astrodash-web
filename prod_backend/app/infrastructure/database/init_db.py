from sqlalchemy import create_engine
from app.config.settings import get_settings
from app.infrastructure.database.models import Base

def init_database():
    """Initialize the database by creating all tables."""
    settings = get_settings()
    database_url = str(settings.db_url) if settings.db_url else "sqlite:///./test.db"

    engine = create_engine(
        database_url,
        connect_args={"check_same_thread": False} if database_url.startswith("sqlite") else {}
    )

    # Create all tables
    Base.metadata.create_all(bind=engine)
    print(f"Database initialized at: {database_url}")

if __name__ == "__main__":
    init_database()
