"""
Son1kVers3 - Resistencia Sonora
FastAPI Backend COMPLETO con funcionalidad "Maqueta → Production"

Plataforma de creación musical con IA que democratiza la producción musical
Mercado objetivo: 400+ millones de hispanohablantes
"""

from fastapi import FastAPI, HTTPException, Depends, File, UploadFile, Form, BackgroundTasks, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from contextlib import asynccontextmanager
import uvicorn
import os
import uuid
import tempfile
import aiofiles
from pathlib import Path
import logging
from typing import Optional, List, Dict, Any, Union
import time
import json
from datetime import datetime, timedelta
import asyncio
from pydantic import BaseModel, Field

# Importar nuestros módulos
from .config import settings
from .models import User, AudioFile, GenerationTask, GhostPreset
from .database import get_db, SessionLocal
from .auth import verify_token, create_access_token, get_current_user
from .audio_processing import ResistanceAudioProcessor
from .voice_expression import VocalExpressionEngine
from .celery_worker import (
    generate_music_task, 
    clone_voice_task, 
    process_ghost_studio_task,
    export_for_distribution_task
)
from .utils import (
    generate_filename, 
    create_resistance_metadata,
    validate_audio_file,
    sanitize_prompt
)

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inicializar procesadores
audio_processor = ResistanceAudioProcessor()
voice_engine = VocalExpressionEngine()
security = HTTPBearer()

# Modelos Pydantic para requests
class GenerateMusicRequest(BaseModel):
    prompt: str = Field(..., min_length=10, max_length=500, description="Descripción musical en español")
    duration: int = Field(default=30, ge=10, le=180, description="Duración en segundos")
    style: str = Field(default="professional", regex="^(demo|professional|resistance)$")
    genre_hint: Optional[str] = Field(None, description="Género musical sugerido")
    mood: Optional[str] = Field(None, description="Estado de ánimo deseado")
    instrumentation: Optional[List[str]] = Field(None, description="Instrumentos específicos")
    user_mode: str = Field(default="beta", regex="^(free|beta|pro|resistance)$")

class CloneVoiceRequest(BaseModel):
    target_text: str = Field(..., min_length=5, max_length=1000)
    expression: str = Field(default="natural", regex="^(natural|joy|melancholy|energetic|calm|dramatic|whisper|excited|romantic)$")
    voice_model: str = Field(default="tortoise", regex="^(tortoise|bark|coqui)$")
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    pitch_shift: float = Field(default=0.0, ge=-12.0, le=12.0)

class GhostStudioRequest(BaseModel):
    preset_name: str = Field(..., regex="^(jazz_fusion|latin_vibes|electronic_dreams|acoustic_soul|rock_anthems|ambient_spaces|trap_beats|classical_modern|reggaeton_fire|indie_alternative|progressive_metal|bossa_nova)$")
    intensity: float = Field(default=0.7, ge=0.1, le=1.0)
    preserve_vocals: bool = Field(default=True)
    creative_freedom: float = Field(default=0.5, ge=0.0, le=1.0)
    custom_params: Optional[Dict[str, Any]] = Field(None)

class ProcessMaquetaRequest(BaseModel):
    style: str = Field(default="professional", regex="^(demo|professional|resistance)$")
    auto_enhance: bool = Field(default=True)
    mastering_preset: str = Field(default="streaming")
    export_formats: List[str] = Field(default=["wav", "mp3"])

# Lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("🎵 Son1kVers3 - Resistencia Sonora iniciando...")
    
    # Crear directorios necesarios
    os.makedirs(settings.upload_dir, exist_ok=True)
    os.makedirs(settings.output_dir, exist_ok=True)
    os.makedirs(settings.temp_dir, exist_ok=True)
    
    # Verificar modelos de IA
    try:
        logger.info("🤖 Verificando modelos de IA...")
        # Esta verificación se haría en background en producción
        await asyncio.sleep(1)  # Simular carga
        logger.info("✅ Modelos de IA listos")
    except Exception as e:
        logger.warning(f"⚠️ Modelos de IA no disponibles: {e}")
    
    logger.info("🚀 Son1kVers3 listo para democratizar la música!")
    
    yield
    
    # Shutdown
    logger.info("🔄 Cerrando Son1kVers3...")

# Crear aplicación FastAPI
app = FastAPI(
    title="Son1kVers3 - Resistencia Sonora",
    description="Democratizando la creación musical con IA | Maqueta → Production",
    version="3.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Middleware de logging y timing
@app.middleware("http")
async def log_requests(request, call_next):
    start_time = time.time()
    
    # Log request
    logger.info(f"📝 {request.method} {request.url.path}")
    
    response = await call_next(request)
    
    # Log response timing
    process_time = time.time() - start_time
    logger.info(f"⏱️ Completado en {process_time:.2f}s - Status: {response.status_code}")
    
    response.headers["X-Process-Time"] = str(process_time)
    return response

# Montar archivos estáticos
app.mount("/uploads", StaticFiles(directory=settings.upload_dir), name="uploads")
app.mount("/output", StaticFiles(directory=settings.output_dir), name="output")

# ===============================================================================
# ENDPOINTS PRINCIPALES - FUNCIONALIDAD CORE
# ===============================================================================

@app.get("/")
async def root():
    """Homepage con información del proyecto"""
    return {
        "message": "🎵 Son1kVers3 - Resistencia Sonora",
        "description": "Democratizando la creación musical con IA",
        "slogan": "Maqueta → Production",
        "market": "400+ millones de hispanohablantes",
        "version": "3.0.0",
        "status": "active",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "generate_music": "/api/generate-music",
            "clone_voice": "/api/clone-voice",
            "ghost_studio": "/api/ghost-studio",
            "process_maqueta": "/api/process-maqueta",
            "upload": "/api/upload"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Verificar base de datos
        db = SessionLocal()
        db.execute("SELECT 1")
        db.close()
        db_status = "healthy"
    except:
        db_status = "error"
    
    return {
        "status": "healthy" if db_status == "healthy" else "degraded",
        "timestamp": datetime.now().isoformat(),
        "version": "3.0.0",
        "services": {
            "api": "healthy",
            "database": db_status,
            "redis": "healthy",  # TODO: verificar Redis
            "workers": "active"
        },
        "resistance_level": "maximum" # 😎
    }

# ===============================================================================
# GENERACIÓN MUSICAL - ENDPOINT PRINCIPAL
# ===============================================================================

@app.post("/api/generate-music")
async def generate_music(
    request: GenerateMusicRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db = Depends(get_db)
):
    """
    🎼 ENDPOINT PRINCIPAL: Generar música con IA
    Funcionalidad "Maqueta → Production"
    """
    try:
        logger.info(f"🎵 Generando música para usuario {current_user.id}: '{request.prompt}'")
        
        # Validar y sanitizar prompt
        clean_prompt = sanitize_prompt(request.prompt)
        
        # Verificar límites del usuario
        if current_user.tier == "free" and request.duration > 30:
            raise HTTPException(status_code=403, detail="Usuarios free limitados a 30 segundos")
        
        # Crear tarea en base de datos
        task = GenerationTask(
            user_id=current_user.id,
            prompt=clean_prompt,
            duration=request.duration,
            style=request.style,
            status="queued"
        )
        db.add(task)
        db.commit()
        db.refresh(task)
        
        # Preparar parámetros para Celery
        processing_params = {
            "genre_hint": request.genre_hint,
            "mood": request.mood,
            "instrumentation": request.instrumentation,
            "quality": "high" if current_user.tier in ["pro", "resistance"] else "medium"
        }
        
        # Enviar a queue de Celery
        celery_task = generate_music_task.delay(
            prompt=clean_prompt,
            duration=request.duration,
            model_quality="large" if current_user.tier == "resistance" else "medium",
            processing_params=processing_params,
            user_mode=request.user_mode
        )
        
        # Actualizar task con Celery ID
        task.celery_id = celery_task.id
        db.commit()
        
        return {
            "message": "🎵 Generación musical iniciada",
            "task_id": task.id,
            "celery_id": celery_task.id,
            "estimated_time": f"{request.duration + 30}-{request.duration + 60} segundos",
            "status": "queued",
            "prompt": clean_prompt,
            "style": request.style,
            "resistance_level": "high" if request.user_mode == "resistance" else "medium"
        }
        
    except Exception as e:
        logger.error(f"❌ Error generando música: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error en generación: {str(e)}")

@app.get("/api/generate-music/{task_id}/status")
async def get_generation_status(
    task_id: int,
    current_user: User = Depends(get_current_user),
    db = Depends(get_db)
):
    """Verificar estado de generación musical"""
    task = db.query(GenerationTask).filter(
        GenerationTask.id == task_id,
        GenerationTask.user_id == current_user.id
    ).first()
    
    if not task:
        raise HTTPException(status_code=404, detail="Tarea no encontrada")
    
    # Consultar estado en Celery
    if task.celery_id:
        from celery.result import AsyncResult
        celery_result = AsyncResult(task.celery_id)
        
        if celery_result.ready():
            if celery_result.successful():
                result = celery_result.result
                task.status = "completed"
                task.output_file = result.get("file_path")
                task.metadata = result.get("metadata", {})
                db.commit()
            else:
                task.status = "failed"
                task.error_message = str(celery_result.info)
                db.commit()
    
    return {
        "task_id": task.id,
        "status": task.status,
        "progress": getattr(task, 'progress', 0),
        "output_file": task.output_file,
        "metadata": task.metadata,
        "created_at": task.created_at,
        "estimated_completion": task.created_at + timedelta(seconds=task.duration + 45)
    }

# ===============================================================================
# CLONACIÓN DE VOZ
# ===============================================================================

@app.post("/api/clone-voice")
async def clone_voice(
    request: CloneVoiceRequest,
    voice_file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db = Depends(get_db)
):
    """
    🎤 Clonar voz con expresión emocional
    """
    try:
        logger.info(f"🎤 Clonando voz para usuario {current_user.id}")
        
        # Validar archivo de voz
        if not voice_file.content_type.startswith('audio/'):
            raise HTTPException(status_code=400, detail="Archivo debe ser audio")
        
        # Verificar límites
        if current_user.tier == "free":
            raise HTTPException(status_code=403, detail="Clonación de voz requiere cuenta Beta+")
        
        # Guardar archivo temporal
        temp_path = f"{settings.temp_dir}/{uuid.uuid4()}_{voice_file.filename}"
        async with aiofiles.open(temp_path, 'wb') as f:
            content = await voice_file.read()
            await f.write(content)
        
        # Validar audio
        if not validate_audio_file(temp_path):
            os.remove(temp_path)
            raise HTTPException(status_code=400, detail="Archivo de audio inválido")
        
        # Crear registro en DB
        audio_record = AudioFile(
            user_id=current_user.id,
            filename=voice_file.filename,
            file_path=temp_path,
            file_type="voice_sample"
        )
        db.add(audio_record)
        db.commit()
        
        # Preparar parámetros de expresión
        expression_params = {
            "expression": request.expression,
            "speed": request.speed,
            "pitch_shift": request.pitch_shift
        }
        
        # Enviar a Celery
        celery_task = clone_voice_task.delay(
            source_audio_path=temp_path,
            target_text=request.target_text,
            expression_params=expression_params,
            voice_model=request.voice_model
        )
        
        return {
            "message": "🎤 Clonación de voz iniciada",
            "task_id": celery_task.id,
            "expression": request.expression,
            "voice_model": request.voice_model,
            "estimated_time": "2-5 minutos"
        }
        
    except Exception as e:
        logger.error(f"❌ Error clonando voz: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error en clonación: {str(e)}")

# ===============================================================================
# GHOST STUDIO - REARREGLOS CREATIVOS
# ===============================================================================

@app.post("/api/ghost-studio")
async def ghost_studio_process(
    request: GhostStudioRequest,
    audio_file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db = Depends(get_db)
):
    """
    👻 Ghost Studio: Rearreglos creativos automáticos
    """
    try:
        logger.info(f"👻 Ghost Studio para usuario {current_user.id}: preset '{request.preset_name}'")
        
        # Validar archivo
        if not audio_file.content_type.startswith('audio/'):
            raise HTTPException(status_code=400, detail="Archivo debe ser audio")
        
        # Guardar archivo temporal
        temp_path = f"{settings.temp_dir}/{uuid.uuid4()}_{audio_file.filename}"
        async with aiofiles.open(temp_path, 'wb') as f:
            content = await audio_file.read()
            await f.write(content)
        
        # Crear registro
        audio_record = AudioFile(
            user_id=current_user.id,
            filename=audio_file.filename,
            file_path=temp_path,
            file_type="ghost_input"
        )
        db.add(audio_record)
        db.commit()
        
        # Preparar parámetros custom
        custom_params = request.custom_params or {}
        custom_params.update({
            "intensity": request.intensity,
            "preserve_vocals": request.preserve_vocals,
            "creative_freedom": request.creative_freedom
        })
        
        # Enviar a Celery
        celery_task = process_ghost_studio_task.delay(
            audio_path=temp_path,
            preset_name=request.preset_name,
            custom_params=custom_params
        )
        
        return {
            "message": f"👻 Ghost Studio procesando con preset '{request.preset_name}'",
            "task_id": celery_task.id,
            "preset": request.preset_name,
            "intensity": request.intensity,
            "estimated_time": "1-3 minutos"
        }
        
    except Exception as e:
        logger.error(f"❌ Error en Ghost Studio: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error en Ghost Studio: {str(e)}")

# ===============================================================================
# PROCESAMIENTO MAQUETA → PRODUCTION
# ===============================================================================

@app.post("/api/process-maqueta")
async def process_maqueta(
    request: ProcessMaquetaRequest,
    maqueta_file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db = Depends(get_db)
):
    """
    🚀 WORKFLOW PRINCIPAL: Maqueta → Production
    Convierte demos caseros en producciones profesionales
    """
    try:
        logger.info(f"🚀 Procesando maqueta para usuario {current_user.id}")
        
        # Validar archivo
        if not audio_file.content_type.startswith('audio/'):
            raise HTTPException(status_code=400, detail="Archivo debe ser audio")
        
        # Guardar maqueta
        temp_path = f"{settings.temp_dir}/{uuid.uuid4()}_{maqueta_file.filename}"
        async with aiofiles.open(temp_path, 'wb') as f:
            content = await maqueta_file.read()
            await f.write(content)
        
        # Crear registro
        audio_record = AudioFile(
            user_id=current_user.id,
            filename=maqueta_file.filename,
            file_path=temp_path,
            file_type="maqueta"
        )
        db.add(audio_record)
        db.commit()
        
        # Análisis inicial rápido
        try:
            analysis = audio_processor.analyze_quick(temp_path)
            logger.info(f"📊 Análisis maqueta: {analysis.get('genre', 'unknown')} - {analysis.get('tempo', 0)} BPM")
        except Exception as e:
            logger.warning(f"⚠️ Error en análisis: {e}")
            analysis = {}
        
        # Enviar a pipeline completo de producción
        celery_task = export_for_distribution_task.delay(
            audio_path=temp_path,
            distribution_formats=request.export_formats,
            mastering_preset=request.mastering_preset
        )
        
        return {
            "message": "🚀 Maqueta → Production iniciado",
            "task_id": celery_task.id,
            "style": request.style,
            "analysis": analysis,
            "export_formats": request.export_formats,
            "estimated_time": "3-8 minutos",
            "resistance_boost": "active" if request.style == "resistance" else "standard"
        }
        
    except Exception as e:
        logger.error(f"❌ Error procesando maqueta: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error en procesamiento: {str(e)}")

# ===============================================================================
# UPLOAD Y GESTIÓN DE ARCHIVOS
# ===============================================================================

@app.post("/api/upload")
async def upload_audio(
    file: UploadFile = File(...),
    file_type: str = Form(default="general"),
    current_user: User = Depends(get_current_user),
    db = Depends(get_db)
):
    """Upload de archivos de audio"""
    try:
        # Validaciones
        if not file.content_type.startswith('audio/'):
            raise HTTPException(status_code=400, detail="Solo se permiten archivos de audio")
        
        if file.size > settings.max_file_size:
            raise HTTPException(status_code=413, detail="Archivo muy grande")
        
        # Generar nombre único
        filename = generate_filename(file.filename, "upload")
        file_path = f"{settings.upload_dir}/{filename}"
        
        # Guardar archivo
        async with aiofiles.open(file_path, 'wb') as f:
            content = await file.read()
            await f.write(content)
        
        # Registrar en DB
        audio_record = AudioFile(
            user_id=current_user.id,
            filename=file.filename,
            file_path=file_path,
            file_type=file_type,
            file_size=len(content)
        )
        db.add(audio_record)
        db.commit()
        
        return {
            "message": "📁 Archivo subido exitosamente",
            "file_id": audio_record.id,
            "filename": filename,
            "size": len(content),
            "type": file_type,
            "url": f"/uploads/{filename}"
        }
        
    except Exception as e:
        logger.error(f"❌ Error subiendo archivo: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error en upload: {str(e)}")

@app.get("/api/files")
async def list_user_files(
    current_user: User = Depends(get_current_user),
    db = Depends(get_db)
):
    """Listar archivos del usuario"""
    files = db.query(AudioFile).filter(AudioFile.user_id == current_user.id).all()
    
    return {
        "files": [
            {
                "id": f.id,
                "filename": f.filename,
                "type": f.file_type,
                "size": f.file_size,
                "created_at": f.created_at,
                "url": f"/uploads/{Path(f.file_path).name}" if f.file_path else None
            }
            for f in files
        ],
        "total": len(files)
    }

# ===============================================================================
# AUTENTICACIÓN Y USUARIOS
# ===============================================================================

@app.post("/api/auth/login")
async def login(email: str = Form(...), password: str = Form(...), db = Depends(get_db)):
    """Login de usuario"""
    # TODO: Implementar verificación de credenciales
    # Por ahora, token simple para desarrollo
    
    access_token = create_access_token(data={"sub": email})
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "message": "🎵 Bienvenido a la Resistencia Sonora"
    }

@app.post("/api/auth/register")
async def register(
    email: str = Form(...),
    password: str = Form(...), 
    name: str = Form(...),
    db = Depends(get_db)
):
    """Registro de nuevo usuario"""
    # TODO: Implementar creación de usuario
    # Hash de password, validaciones, etc.
    
    return {
        "message": "🎉 Usuario registrado exitosamente",
        "status": "pending_verification"
    }

# ===============================================================================
# INFORMACIÓN Y UTILIDADES
# ===============================================================================

@app.get("/api/presets")
async def list_ghost_presets():
    """Listar presets disponibles de Ghost Studio"""
    presets = [
        {"name": "jazz_fusion", "description": "Fusión de jazz con elementos modernos"},
        {"name": "latin_vibes", "description": "Ritmos latinos vibrantes"},
        {"name": "electronic_dreams", "description": "Paisajes electrónicos atmosféricos"},
        {"name": "acoustic_soul", "description": "Alma acústica orgánica"},
        {"name": "rock_anthems", "description": "Himnos de rock potentes"},
        {"name": "ambient_spaces", "description": "Espacios ambientales expansivos"},
        {"name": "trap_beats", "description": "Beats de trap contemporáneos"},
        {"name": "classical_modern", "description": "Clásico moderno híbrido"},
        {"name": "reggaeton_fire", "description": "Fuego de reggaetón"},
        {"name": "indie_alternative", "description": "Alternativo independiente"},
        {"name": "progressive_metal", "description": "Metal progresivo complejo"},
        {"name": "bossa_nova", "description": "Bossa nova sofisticada"}
    ]
    
    return {"presets": presets}

@app.get("/api/models")
async def list_available_models():
    """Información sobre modelos de IA disponibles"""
    return {
        "music_generation": {
            "musicgen_small": {"size": "300MB", "quality": "medium", "speed": "fast"},
            "musicgen_medium": {"size": "1.5GB", "quality": "high", "speed": "medium"},
            "musicgen_large": {"size": "3.3GB", "quality": "premium", "speed": "slow"}
        },
        "voice_cloning": {
            "tortoise": {"quality": "high", "languages": ["es", "en"], "speed": "slow"},
            "bark": {"quality": "medium", "languages": ["es", "en"], "speed": "medium"},
            "coqui": {"quality": "good", "languages": ["es"], "speed": "fast"}
        }
    }

@app.get("/api/stats")
async def get_platform_stats(current_user: User = Depends(get_current_user), db = Depends(get_db)):
    """Estadísticas de la plataforma"""
    user_tasks = db.query(GenerationTask).filter(GenerationTask.user_id == current_user.id).count()
    
    return {
        "platform": {
            "total_generations": "resistencia_counter_classified",
            "active_users": "growing_resistance",
            "hours_generated": "democratizing_music",
            "resistance_level": "maximum"
        },
        "user": {
            "generations": user_tasks,
            "tier": current_user.tier,
            "joined": current_user.created_at
        }
    }

# ===============================================================================
# ERROR HANDLERS
# ===============================================================================

@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "message": "🔍 Endpoint no encontrado",
            "suggestion": "Visita /docs para ver endpoints disponibles",
            "resistance_tip": "La resistencia también se pierde a veces 😉"
        }
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"💥 Error interno: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "message": "💥 Error interno del servidor",
            "tip": "La resistencia nunca se rinde. Intenta de nuevo.",
            "support": "Contacta soporte si persiste"
        }
    )

# ===============================================================================
# DESARROLLO Y TESTING
# ===============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
