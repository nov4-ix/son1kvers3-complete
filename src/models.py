"""
Database configuration and models for Son1k v3.0
Supports both SQLite (development) and PostgreSQL (production)
"""

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, Boolean, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.sql import func
from contextlib import contextmanager
import logging
from typing import Generator
from datetime import datetime

from src.core.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)

# Database engine
if settings.DATABASE_URL.startswith("sqlite"):
    # SQLite configuration for development
    engine = create_engine(
        settings.DATABASE_URL,
        connect_args={"check_same_thread": False},  # SQLite specific
        echo=settings.APP_DEBUG
    )
else:
    # PostgreSQL configuration for production
    engine = create_engine(
        settings.DATABASE_URL,
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,
        echo=settings.APP_DEBUG
    )

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base model class
Base = declarative_base()

# Database dependency
def get_db() -> Generator[Session, None, None]:
    """Database session dependency for FastAPI"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@contextmanager
def get_db_session():
    """Context manager for database sessions"""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

# === MODELS ===

class User(Base):
    """User model for authentication and session management"""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    username = Column(String(100), unique=True, index=True, nullable=True)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255), nullable=True)
    is_active = Column(Boolean, default=True)
    is_superuser = Column(Boolean, default=False)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    last_login = Column(DateTime(timezone=True), nullable=True)
    
    # User preferences
    preferences = Column(JSON, default={})
    
    def __repr__(self):
        return f"<User(id={self.id}, email='{self.email}')>"

class GenerationJob(Base):
    """Model for tracking music generation jobs"""
    __tablename__ = "generation_jobs"
    
    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(String(36), unique=True, index=True, nullable=False)  # UUID
    user_id = Column(Integer, nullable=True)  # Optional user association
    
    # Job details
    job_type = Column(String(50), nullable=False)  # 'manual', 'ghost', 'maqueta'
    status = Column(String(20), default="queued")  # queued, running, done, error
    
    # Generation parameters
    prompt = Column(Text, nullable=False)
    duration = Column(Float, nullable=False)
    model_name = Column(String(100), nullable=True)
    parameters = Column(JSON, default={})
    
    # Results
    output_url = Column(String(500), nullable=True)
    output_filename = Column(String(255), nullable=True)
    generation_time = Column(Float, nullable=True)
    error_message = Column(Text, nullable=True)
    
    # Metadata
    device_used = Column(String(50), nullable=True)
    audio_stats = Column(JSON, default={})
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    
    def __repr__(self):
        return f"<GenerationJob(id={self.id}, job_id='{self.job_id}', status='{self.status}')>"

class MaquetaSession(Base):
    """Model for Maqueta → Production sessions"""
    __tablename__ = "maqueta_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(36), unique=True, index=True, nullable=False)  # UUID
    user_id = Column(Integer, nullable=True)
    
    # Input files
    demo_filename = Column(String(255), nullable=False)
    demo_url = Column(String(500), nullable=False)
    demo_size_mb = Column(Float, nullable=True)
    
    # User input
    prompt_original = Column(Text, nullable=False)
    prompt_final = Column(Text, nullable=True)  # AI-enhanced prompt
    target_duration = Column(Float, nullable=False)
    
    # Analysis results
    analysis_data = Column(JSON, default={})
    
    # Processing parameters
    processing_params = Column(JSON, default={})
    
    # Output
    production_filename = Column(String(255), nullable=True)
    production_url = Column(String(500), nullable=True)
    processing_chain = Column(JSON, default=[])
    
    # Performance metrics
    processing_time = Column(Float, nullable=True)
    analysis_time = Column(Float, nullable=True)
    generation_time = Column(Float, nullable=True)
    postprocess_time = Column(Float, nullable=True)
    
    # Status
    status = Column(String(20), default="processing")  # processing, completed, failed
    error_message = Column(Text, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True), nullable=True)
    
    def __repr__(self):
        return f"<MaquetaSession(id={self.id}, session_id='{self.session_id}', status='{self.status}')>"

class AudioFile(Base):
    """Model for tracking generated audio files"""
    __tablename__ = "audio_files"
    
    id = Column(Integer, primary_key=True, index=True)
    file_id = Column(String(36), unique=True, index=True, nullable=False)  # UUID
    
    # File information
    filename = Column(String(255), nullable=False)
    original_filename = Column(String(255), nullable=True)
    file_path = Column(String(500), nullable=False)
    file_size_mb = Column(Float, nullable=False)
    mime_type = Column(String(100), nullable=False)
    
    # Audio properties
    duration_seconds = Column(Float, nullable=False)
    sample_rate = Column(Integer, nullable=False)
    channels = Column(Integer, default=1)
    bitrate = Column(Integer, nullable=True)
    
    # Metadata
    audio_format = Column(String(20), nullable=False)  # wav, mp3, etc.
    peak_level = Column(Float, nullable=True)
    rms_level = Column(Float, nullable=True)
    lufs_level = Column(Float, nullable=True)
    
    # Relationships
    generation_job_id = Column(String(36), nullable=True)
    maqueta_session_id = Column(String(36), nullable=True)
    user_id = Column(Integer, nullable=True)
    
    # File management
    is_temporary = Column(Boolean, default=False)
    expires_at = Column(DateTime(timezone=True), nullable=True)
    download_count = Column(Integer, default=0)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    last_accessed = Column(DateTime(timezone=True), nullable=True)
    
    def __repr__(self):
        return f"<AudioFile(id={self.id}, filename='{self.filename}')>"

class SystemMetrics(Base):
    """Model for tracking system performance metrics"""
    __tablename__ = "system_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Metrics
    metric_type = Column(String(50), nullable=False)  # generation_time, memory_usage, etc.
    value = Column(Float, nullable=False)
    unit = Column(String(20), nullable=True)  # seconds, MB, etc.
    
    # Context
    context = Column(JSON, default={})  # Additional context data
    device = Column(String(50), nullable=True)
    model_name = Column(String(100), nullable=True)
    
    # Timestamp
    recorded_at = Column(DateTime(timezone=True), server_default=func.now())
    
    def __repr__(self):
        return f"<SystemMetrics(id={self.id}, type='{self.metric_type}', value={self.value})>"

class APIUsage(Base):
    """Model for tracking API usage and rate limiting"""
    __tablename__ = "api_usage"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Request details
    endpoint = Column(String(200), nullable=False)
    method = Column(String(10), nullable=False)
    status_code = Column(Integer, nullable=False)
    response_time_ms = Column(Float, nullable=False)
    
    # User context
    user_id = Column(Integer, nullable=True)
    ip_address = Column(String(45), nullable=True)  # IPv6 compatible
    user_agent = Column(Text, nullable=True)
    
    # Rate limiting
    requests_count = Column(Integer, default=1)
    
    # Timestamp
    requested_at = Column(DateTime(timezone=True), server_default=func.now())
    
    def __repr__(self):
        return f"<APIUsage(id={self.id}, endpoint='{self.endpoint}', status={self.status_code})>"

# === DATABASE OPERATIONS ===

def create_tables():
    """Create all database tables"""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Failed to create database tables: {e}")
        raise

def drop_tables():
    """Drop all database tables (use with caution!)"""
    try:
        Base.metadata.drop_all(bind=engine)
        logger.info("Database tables dropped successfully")
    except Exception as e:
        logger.error(f"Failed to drop database tables: {e}")
        raise

def get_database_info():
    """Get database information and statistics"""
    try:
        with get_db_session() as db:
            info = {
                "engine": str(engine.url),
                "dialect": engine.dialect.name,
                "tables": [],
                "total_records": 0
            }
            
            # Get table information
            for table_name in Base.metadata.tables.keys():
                table = Base.metadata.tables[table_name]
                try:
                    count = db.execute(f"SELECT COUNT(*) FROM {table_name}").scalar()
                    info["tables"].append({
                        "name": table_name,
                        "columns": len(table.columns),
                        "records": count
                    })
                    info["total_records"] += count
                except Exception as e:
                    logger.warning(f"Failed to get info for table {table_name}: {e}")
            
            return info
    except Exception as e:
        logger.error(f"Failed to get database info: {e}")
        return {"error": str(e)}

def cleanup_expired_files():
    """Clean up expired temporary files"""
    try:
        with get_db_session() as db:
            # Find expired files
            expired_files = db.query(AudioFile).filter(
                AudioFile.is_temporary == True,
                AudioFile.expires_at < datetime.utcnow()
            ).all()
            
            cleaned_count = 0
            for file_record in expired_files:
                try:
                    # Delete file from filesystem
                    from pathlib import Path
                    file_path = Path(file_record.file_path)
                    if file_path.exists():
                        file_path.unlink()
                    
                    # Remove from database
                    db.delete(file_record)
                    cleaned_count += 1
                    
                except Exception as e:
                    logger.warning(f"Failed to cleanup file {file_record.filename}: {e}")
            
            logger.info(f"Cleaned up {cleaned_count} expired files")
            return cleaned_count
            
    except Exception as e:
        logger.error(f"Failed to cleanup expired files: {e}")
        return 0

def get_usage_statistics(days: int = 30):
    """Get usage statistics for the last N days"""
    try:
        from datetime import timedelta
        
        with get_db_session() as db:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            # Generation jobs stats
            jobs_stats = db.query(GenerationJob).filter(
                GenerationJob.created_at >= cutoff_date
            ).all()
            
            # Maqueta sessions stats
            sessions_stats = db.query(MaquetaSession).filter(
                MaquetaSession.created_at >= cutoff_date
            ).all()
            
            # API usage stats
            api_stats = db.query(APIUsage).filter(
                APIUsage.requested_at >= cutoff_date
            ).all()
            
            return {
                "period_days": days,
                "generation_jobs": {
                    "total": len(jobs_stats),
                    "by_type": {},
                    "by_status": {},
                    "avg_duration": 0
                },
                "maqueta_sessions": {
                    "total": len(sessions_stats),
                    "successful": len([s for s in sessions_stats if s.status == "completed"]),
                    "avg_processing_time": 0
                },
                "api_usage": {
                    "total_requests": len(api_stats),
                    "unique_endpoints": len(set(req.endpoint for req in api_stats)),
                    "avg_response_time": sum(req.response_time_ms for req in api_stats) / len(api_stats) if api_stats else 0
                }
            }
            
    except Exception as e:
        logger.error(f"Failed to get usage statistics: {e}")
        return {"error": str(e)}

# === INITIALIZATION ===

def init_database():
    """Initialize database with default data if needed"""
    try:
        create_tables()
        
        # Add any default data here if needed
        with get_db_session() as db:
            # Check if we need to create default admin user, etc.
            pass
            
        logger.info("Database initialized successfully")
        
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise

# Health check function
def check_database_health():
    """Check database connectivity and health"""
    try:
        with get_db_session() as db:
            # Simple query to test connectivity
            db.execute("SELECT 1").scalar()
            
        return {
            "status": "healthy",
            "engine": engine.dialect.name,
            "connection_pool": {
                "size": engine.pool.size(),
                "checked_out": engine.pool.checkedout(),
                "overflow": engine.pool.overflow()
            } if hasattr(engine.pool, 'size') else "N/A"
        }
        
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }