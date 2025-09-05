"""
Configuration settings for Son1k v3.0
Handles environment variables and application settings
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List, Optional
import os
from pathlib import Path

class Settings(BaseSettings):
    """Application settings from environment variables"""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )
    
    # === APPLICATION ===
    APP_NAME: str = "Son1k v3.0 API"
    APP_VERSION: str = "3.0.0"
    APP_DEBUG: bool = False
    API_V1_STR: str = "/api/v1"
    
    # === SECURITY ===
    SECRET_KEY: str = "944af7a84933a3572e801f641890b2021ec7703b1dce00966c8b996ed59eb7c"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    
    # === DATABASE ===
    DATABASE_URL: str = "sqlite:///./storage/son1k.db"
    
    # === CORS ===
    BACKEND_CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000", 
        "http://127.0.0.1:5173"
    ]
    
    # === AUDIO & AI ===
    MUSICGEN_MODEL: str = "facebook/musicgen-small"
    MAX_AUDIO_DURATION: int = 30  # seconds
    MAX_FILE_SIZE: int = 100 * 1024 * 1024  # 100MB
    SUPPORTED_FORMATS: List[str] = [".wav", ".mp3", ".flac", ".aiff", ".m4a"]
    
    # Audio processing settings
    SAMPLE_RATE: int = 32000
    AUDIO_CHANNELS: int = 1  # mono
    
    # === STORAGE ===
    STORAGE_ROOT: str = "storage"
    UPLOADS_DIR: str = "storage/uploads"
    OUTPUT_DIR: str = "storage/output"
    MODELS_DIR: str = "storage/models"
    
    # === PROCESSING ===
    MAX_CONCURRENT_JOBS: int = 2
    JOB_TIMEOUT_SECONDS: int = 300  # 5 minutes
    CLEANUP_INTERVAL_HOURS: int = 24
    
    # === GHOST STUDIO ===
    GHOST_PRESETS_FILE: str = "backend/data/presets.json"
    GHOST_JOBS_FILE: str = "backend/data/jobs.json"
    GHOST_MAX_JOBS: int = 100
    
    # === AUDIO ANALYSIS ===
    ANALYSIS_SETTINGS: dict = {
        "tempo_min_bpm": 60,
        "tempo_max_bpm": 200,
        "hop_length": 512,
        "frame_length": 2048,
        "fft_size": 4096
    }
    
    # === POSTPROCESSING ===
    SSL_EQ_SETTINGS: dict = {
        "low_freq": 80,      # Hz
        "mid1_freq": 400,    # Hz  
        "mid2_freq": 3000,   # Hz
        "high_freq": 8000,   # Hz
        "hpf_freq": 20       # Hz
    }
    
    NEVE_SATURATION_SETTINGS: dict = {
        "oversample_factor": 4,
        "harmonic_2nd": 0.1,
        "harmonic_3rd": 0.05
    }
    
    MASTERING_SETTINGS: dict = {
        "lufs_target": -14.0,
        "ceiling_db": -0.3,
        "fade_in_ms": 50,
        "fade_out_ms": 200
    }
    
    # === LOGGING ===
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # === RATE LIMITING ===
    RATE_LIMIT_PER_MINUTE: int = 60
    RATE_LIMIT_PER_HOUR: int = 1000
    
    @property
    def storage_paths(self) -> dict:
        """Get all storage paths as Path objects"""
        return {
            "root": Path(self.STORAGE_ROOT),
            "uploads": Path(self.UPLOADS_DIR),
            "output": Path(self.OUTPUT_DIR), 
            "models": Path(self.MODELS_DIR),
            "ghost_uploads": Path(self.UPLOADS_DIR) / "ghost",
            "ghost_output": Path(self.OUTPUT_DIR) / "ghost"
        }
    
    @property
    def is_development(self) -> bool:
        """Check if running in development mode"""
        return self.APP_DEBUG or os.getenv("ENVIRONMENT") == "development"
    
    @property
    def is_production(self) -> bool:
        """Check if running in production mode"""
        return not self.is_development
    
    def get_cors_origins(self) -> List[str]:
        """Get CORS origins, supporting both string and list formats"""
        if isinstance(self.BACKEND_CORS_ORIGINS, str):
            return [origin.strip() for origin in self.BACKEND_CORS_ORIGINS.split(",")]
        return self.BACKEND_CORS_ORIGINS
    
    def validate_file_upload(self, filename: str, file_size: int) -> tuple[bool, str]:
        """Validate uploaded file"""
        # Check file extension
        file_ext = Path(filename).suffix.lower()
        if file_ext not in self.SUPPORTED_FORMATS:
            return False, f"Unsupported format. Use: {', '.join(self.SUPPORTED_FORMATS)}"
        
        # Check file size
        if file_size > self.MAX_FILE_SIZE:
            max_mb = self.MAX_FILE_SIZE / (1024 * 1024)
            return False, f"File too large. Maximum size: {max_mb:.1f}MB"
        
        return True, "File is valid"

# Global settings instance
_settings: Optional[Settings] = None

def get_settings() -> Settings:
    """Get or create settings instance (singleton pattern)"""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings

# Development settings override
class DevelopmentSettings(Settings):
    """Settings for development environment"""
    APP_DEBUG: bool = True
    LOG_LEVEL: str = "DEBUG"
    MUSICGEN_MODEL: str = "facebook/musicgen-small"  # Use smallest model for dev
    MAX_CONCURRENT_JOBS: int = 1  # Limit for development

# Production settings override  
class ProductionSettings(Settings):
    """Settings for production environment"""
    APP_DEBUG: bool = False
    LOG_LEVEL: str = "INFO"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    RATE_LIMIT_PER_MINUTE: int = 30  # More restrictive
    RATE_LIMIT_PER_HOUR: int = 500

def get_settings_for_environment(env: str = None) -> Settings:
    """Get settings for specific environment"""
    if env is None:
        env = os.getenv("ENVIRONMENT", "development")
    
    if env == "production":
        return ProductionSettings()
    elif env == "development":
        return DevelopmentSettings()
    else:
        return Settings()

# Export commonly used settings
def get_storage_paths() -> dict:
    """Get storage paths dictionary"""
    return get_settings().storage_paths

def get_audio_settings() -> dict:
    """Get audio processing settings"""
    settings = get_settings()
    return {
        "sample_rate": settings.SAMPLE_RATE,
        "channels": settings.AUDIO_CHANNELS,
        "max_duration": settings.MAX_AUDIO_DURATION,
        "supported_formats": settings.SUPPORTED_FORMATS
    }