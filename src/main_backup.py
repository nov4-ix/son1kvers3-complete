"""
Son1k v3.0 - AI Music Generation Platform
FastAPI Backend with Maqueta → Production workflow

Main application entry point with all routes and middleware.
"""

from fastapi import FastAPI, HTTPException, Depends, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import uvicorn
import os
from pathlib import Path
import logging
from typing import Optional, List, Dict, Any
import time
import json

# Local imports
from src.core.config import get_settings
from src.core.database import engine, create_tables
from src.api.auth import router as auth_router
from src.api.generate import router as generate_router
from src.api.ghost import router as ghost_router
from src.services.musicgen import MusicGenService
from src.utils.file_utils import setup_directories

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global services
musicgen_service = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan with startup and shutdown events"""
    
    # Startup
    logger.info("🚀 Starting Son1k v3.0...")
    
    # Setup directories
    setup_directories()
    
    # Create database tables
    create_tables()
    
    # Initialize MusicGen service
    global musicgen_service
    musicgen_service = MusicGenService()
    
    # Pre-load model in background (optional)
    try:
        logger.info("⏳ Pre-loading MusicGen model...")
        await musicgen_service.load_model()
        logger.info("✅ MusicGen model loaded successfully")
    except Exception as e:
        logger.warning(f"⚠️ Model pre-loading failed: {e}")
    
    logger.info("✅ Son1k v3.0 startup complete")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down Son1k v3.0...")
    
    # Cleanup
    if musicgen_service:
        musicgen_service.cleanup()
    
    logger.info("✅ Shutdown complete")

# Create FastAPI app
settings = get_settings()
app = FastAPI(
    title="Son1k v3.0 API",
    description="AI Music Generation Platform with Maqueta → Production workflow",
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request logging middleware
@app.middleware("http")
async def log_requests(request, call_next):
    """Log all HTTP requests"""
    start_time = time.time()
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    logger.info(
        f"{request.method} {request.url.path} - "
        f"{response.status_code} - {process_time:.3f}s"
    )
    
    return response

# Static file serving
app.mount("/uploads", StaticFiles(directory="storage/uploads"), name="uploads")
app.mount("/output", StaticFiles(directory="storage/output"), name="output")

# API Routes
app.include_router(auth_router, prefix="/api/v1/auth", tags=["Authentication"])
app.include_router(generate_router, prefix="/api/v1", tags=["Generation"])
app.include_router(ghost_router, prefix="/api/v1/ghost", tags=["Ghost Studio"])

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint with system information"""
    try:
        import torch
        import librosa
        import numpy as np
        from src.services.audio_analysis import AudioAnalyzer
        
        # Check GPU availability
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        
        # Check if model is loaded
        model_loaded = musicgen_service.model is not None if musicgen_service else False
        
        # Check storage directories
        storage_info = {}
        for dir_name in ["uploads", "output", "models"]:
            dir_path = Path(f"storage/{dir_name}")
            storage_info[dir_name] = {
                "exists": dir_path.exists(),
                "files": len(list(dir_path.glob("*"))) if dir_path.exists() else 0
            }
        
        return {
            "status": "healthy",
            "service": "Son1k v3.0 API",
            "version": "3.0.0",
            "timestamp": time.time(),
            "system": {
                "device": device,
                "model_loaded": model_loaded,
                "torch_version": torch.__version__,
                "librosa_available": True,
                "numpy_version": np.__version__
            },
            "storage": storage_info,
            "features": [
                "manual_generation",
                "ghost_studio", 
                "maqueta_production",
                "audio_analysis",
                "professional_postprocessing"
            ]
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

# Models information endpoint
@app.get("/api/v1/models")
async def get_models():
    """Get available MusicGen models information"""
    models = [
        {
            "name": "facebook/musicgen-small",
            "description": "Lightweight model, faster generation",
            "parameters": "300M",
            "memory_usage": "~2GB",
            "speed": "Fast",
            "quality": "Good",
            "recommended": True
        },
        {
            "name": "facebook/musicgen-medium", 
            "description": "Balanced model, good quality",
            "parameters": "1.5B",
            "memory_usage": "~8GB", 
            "speed": "Medium",
            "quality": "Better",
            "recommended": False
        },
        {
            "name": "facebook/musicgen-large",
            "description": "High quality model, slower generation",
            "parameters": "3.3B",
            "memory_usage": "~16GB",
            "speed": "Slow", 
            "quality": "Best",
            "recommended": False
        }
    ]
    
    current_model = musicgen_service.model_name if musicgen_service else "Not loaded"
    
    return {
        "models": models,
        "current_model": current_model,
        "device": musicgen_service.device if musicgen_service else "Unknown"
    }

# Cache management endpoint
@app.delete("/api/v1/cache")
async def clear_cache():
    """Clear model cache and temporary files"""
    try:
        # Clear MusicGen cache
        if musicgen_service:
            musicgen_service.clear_cache()
        
        # Clear PyTorch cache
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Clean temporary files
        import tempfile
        import shutil
        temp_dir = Path(tempfile.gettempdir())
        for temp_file in temp_dir.glob("son1k_*"):
            try:
                if temp_file.is_file():
                    temp_file.unlink()
                elif temp_file.is_dir():
                    shutil.rmtree(temp_file)
            except Exception as e:
                logger.warning(f"Failed to clean {temp_file}: {e}")
        
        return {
            "ok": True,
            "message": "Cache cleared successfully",
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Cache clear failed: {e}")
        raise HTTPException(status_code=500, detail=f"Cache clear failed: {str(e)}")

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"detail": "Resource not found"}
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

# Dependency to get MusicGen service
def get_musicgen_service() -> MusicGenService:
    """Dependency to get MusicGen service instance"""
    if musicgen_service is None:
        raise HTTPException(status_code=503, detail="MusicGen service not available")
    return musicgen_service

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "service": "Son1k v3.0 API",
        "version": "3.0.0",
        "description": "AI Music Generation Platform with Maqueta → Production workflow",
        "features": [
            "🎤 Maqueta → Production workflow",
            "🤖 Ghost Studio automation", 
            "🎛️ Manual music generation",
            "📊 Advanced audio analysis",
            "🎵 Professional postprocessing"
        ],
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "generation": "/api/v1/generate",
            "ghost_studio": "/api/v1/ghost",
            "models": "/api/v1/models"
        },
        "storage": {
            "uploads": "/uploads",
            "output": "/output"
        }
    }

# Development server
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )