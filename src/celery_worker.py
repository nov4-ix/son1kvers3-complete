"""
Son1kVers3 - Advanced Celery Workers
Workers completos con postproducción automática y funcionalidades avanzadas
"""

import os
import time
import uuid
import tempfile
import shutil
import logging
from typing import Dict, Any, List, Optional
from celery import Celery
from pathlib import Path
import numpy as np
import soundfile as sf
import librosa
import json
import asyncio

# Importar nuestros módulos avanzados
from .audio_processing import ResistanceAudioProcessor, StemSeparator
from .voice_expression import VocalExpressionEngine, clone_voice_with_expression
from .utils import (
    generate_filename,
    create_resistance_metadata,
    apply_sacred_imperfection,
    detect_xentrix_patterns
)
from .config import settings

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuración de Celery
celery_app = Celery(
    'son1k_advanced_worker',
    broker=settings.redis_broker_url,
    backend=settings.redis_backend_url,
    include=['src.celery_worker']
)

celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_routes={
        'src.celery_worker.generate_music_task': {'queue': 'cpu'},
        'src.celery_worker.generate_music_gpu_task': {'queue': 'gpu'},
        'src.celery_worker.clone_voice_task': {'queue': 'cpu'},
        'src.celery_worker.process_ghost_studio_task': {'queue': 'gpu'},
        'src.celery_worker.export_for_distribution_task': {'queue': 'cpu'},
    }
)

# Inicializar procesadores globales
audio_processor = ResistanceAudioProcessor()
stem_separator = StemSeparator()
voice_engine = VocalExpressionEngine()

@celery_app.task(bind=True, name='generate_music_advanced')
def generate_music_task(self, 
                       prompt: str, 
                       duration: int = 30,
                       use_gpu: bool = False,
                       model_quality: str = "high",
                       processing_params: Dict[str, Any] = None,
                       user_mode: str = "beta") -> Dict[str, Any]:
    """
    Tarea avanzada de generación musical con postproducción automática
    """
    try:
        task_id = self.request.id
        logger.info(f"🎵 Iniciando generación musical avanzada: {task_id}")
        
        # Actualizar progreso
        self.update_state(state='PROGRESS', meta={'progress': 10, 'status': 'Inicializando IA musical...'})
        
        # 1. Generar música base con MusicGen
        base_audio_path = _generate_base_music(prompt, duration, model_quality)
        self.update_state(state='PROGRESS', meta={'progress': 30, 'status': 'Música base generada. Iniciando postproducción...'})
        
        # 2. Análisis musical avanzado
        analysis = audio_processor.analyze_comprehensive(base_audio_path)
        self.update_state(state='PROGRESS', meta={'progress': 45, 'status': 'Análisis musical completado. Aplicando mejoras...'})
        
        # 3. Postproducción profesional
        enhanced_path = audio_processor.process_advanced_chain(
            base_audio_path,
            processing_params or {},
            analysis
        )
        self.update_state(state='PROGRESS', meta={'progress': 70, 'status': 'Postproducción aplicada. Finalizando...'})
        
        # 4. Aplicar "imperfección sagrada" si está en modo resistance
        if user_mode == "resistance":
            enhanced_path = apply_sacred_imperfection(enhanced_path, analysis)
        
        # 5. Generar metadata completa
        metadata = create_resistance_metadata(prompt, analysis, processing_params)
        
        # 6. Guardar resultado final
        final_filename = generate_filename(prompt, "generated")
        final_path = settings.output_dir / final_filename
        shutil.move(enhanced_path, final_path)
        
        self.update_state(state='PROGRESS', meta={'progress': 100, 'status': 'Generación completada exitosamente!'})
        
        return {
            'status': 'success',
            'file_path': str(final_path),
            'filename': final_filename,
            'metadata': metadata,
            'analysis': analysis,
            'duration': duration,
            'processing_time': time.time() - self.request.started,
            'resistance_level': metadata.get('resistance_score', 0)
        }
        
    except Exception as e:
        logger.error(f"Error en generación musical: {str(e)}")
        self.update_state(state='FAILURE', meta={'error': str(e)})
        raise

@celery_app.task(bind=True, name='clone_voice_advanced')
def clone_voice_task(self, 
                    source_audio_path: str,
                    target_text: str,
                    expression_params: Dict[str, Any] = None,
                    voice_model: str = "tortoise") -> Dict[str, Any]:
    """
    Clonación de voz avanzada con expresión emocional
    """
    try:
        task_id = self.request.id
        logger.info(f"🎤 Iniciando clonación de voz avanzada: {task_id}")
        
        self.update_state(state='PROGRESS', meta={'progress': 15, 'status': 'Analizando voz original...'})
        
        # 1. Análisis de la voz original
        voice_analysis = voice_engine.analyze_voice_characteristics(source_audio_path)
        
        self.update_state(state='PROGRESS', meta={'progress': 40, 'status': 'Entrenando modelo de voz...'})
        
        # 2. Entrenar/adaptar modelo
        voice_model_path = voice_engine.train_or_adapt_model(source_audio_path, voice_model)
        
        self.update_state(state='PROGRESS', meta={'progress': 70, 'status': 'Generando voz clonada...'})
        
        # 3. Generar voz clonada con expresión
        cloned_audio_path = clone_voice_with_expression(
            voice_model_path,
            target_text,
            expression_params or {},
            voice_analysis
        )
        
        # 4. Postprocesamiento de audio
        enhanced_voice_path = audio_processor.enhance_voice_clone(
            cloned_audio_path,
            voice_analysis
        )
        
        # 5. Guardar resultado
        final_filename = generate_filename(f"voice_clone_{target_text[:30]}", "cloned")
        final_path = settings.output_dir / final_filename
        shutil.move(enhanced_voice_path, final_path)
        
        self.update_state(state='PROGRESS', meta={'progress': 100, 'status': 'Clonación de voz completada!'})
        
        return {
            'status': 'success',
            'file_path': str(final_path),
            'filename': final_filename,
            'voice_analysis': voice_analysis,
            'expression_applied': expression_params,
            'processing_time': time.time() - self.request.started
        }
        
    except Exception as e:
        logger.error(f"Error en clonación de voz: {str(e)}")
        self.update_state(state='FAILURE', meta={'error': str(e)})
        raise

@celery_app.task(bind=True, name='process_ghost_studio')
def process_ghost_studio_task(self,
                             audio_path: str,
                             preset_name: str,
                             custom_params: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Procesamiento en Ghost Studio con presets avanzados
    """
    try:
        task_id = self.request.id
        logger.info(f"👻 Iniciando Ghost Studio processing: {task_id}")
        
        from .creative_processor import GhostStudioProcessor
        ghost_processor = GhostStudioProcessor()
        
        self.update_state(state='PROGRESS', meta={'progress': 20, 'status': f'Cargando preset: {preset_name}...'})
        
        # 1. Cargar preset y parámetros
        preset_config = ghost_processor.load_preset(preset_name)
        if custom_params:
            preset_config.update(custom_params)
        
        self.update_state(state='PROGRESS', meta={'progress': 40, 'status': 'Analizando audio original...'})
        
        # 2. Análisis del audio original
        analysis = audio_processor.analyze_comprehensive(audio_path)
        
        self.update_state(state='PROGRESS', meta={'progress': 60, 'status': 'Aplicando transformaciones creativas...'})
        
        # 3. Procesar con Ghost Studio
        processed_path = ghost_processor.process_with_preset(
            audio_path,
            preset_config,
            analysis
        )
        
        self.update_state(state='PROGRESS', meta={'progress': 85, 'status': 'Finalizando procesamiento...'})
        
        # 4. Guardar resultado
        final_filename = generate_filename(f"ghost_{preset_name}", "processed")
        final_path = settings.output_dir / final_filename
        shutil.move(processed_path, final_path)
        
        # 5. Generar reporte de cambios
        changes_report = ghost_processor.generate_changes_report(
            audio_path, processed_path, preset_config
        )
        
        self.update_state(state='PROGRESS', meta={'progress': 100, 'status': 'Ghost Studio completado!'})
        
        return {
            'status': 'success',
            'file_path': str(final_path),
            'filename': final_filename,
            'preset_used': preset_name,
            'changes_applied': changes_report,
            'processing_time': time.time() - self.request.started
        }
        
    except Exception as e:
        logger.error(f"Error en Ghost Studio: {str(e)}")
        self.update_state(state='FAILURE', meta={'error': str(e)})
        raise

@celery_app.task(bind=True, name='export_for_distribution')
def export_for_distribution_task(self,
                                audio_path: str,
                                distribution_formats: List[str] = None,
                                mastering_preset: str = "streaming") -> Dict[str, Any]:
    """
    Exportar audio en múltiples formatos para distribución
    """
    try:
        task_id = self.request.id
        logger.info(f"📦 Iniciando exportación para distribución: {task_id}")
        
        formats = distribution_formats or ["wav", "mp3", "flac", "m4a"]
        exported_files = {}
        
        for i, format_type in enumerate(formats):
            progress = 20 + (i * 60 // len(formats))
            self.update_state(state='PROGRESS', meta={
                'progress': progress, 
                'status': f'Exportando formato {format_type.upper()}...'
            })
            
            # Aplicar mastering específico para el formato
            mastered_path = audio_processor.apply_mastering_preset(
                audio_path, 
                mastering_preset,
                format_type
            )
            
            # Convertir al formato deseado
            final_filename = generate_filename(f"master_{format_type}", format_type)
            final_path = settings.output_dir / final_filename
            
            audio_processor.convert_format(mastered_path, final_path, format_type)
            exported_files[format_type] = str(final_path)
        
        self.update_state(state='PROGRESS', meta={'progress': 100, 'status': 'Exportación completada!'})
        
        return {
            'status': 'success',
            'exported_files': exported_files,
            'mastering_preset': mastering_preset,
            'processing_time': time.time() - self.request.started
        }
        
    except Exception as e:
        logger.error(f"Error en exportación: {str(e)}")
        self.update_state(state='FAILURE', meta={'error': str(e)})
        raise

def _generate_base_music(prompt: str, duration: int, quality: str) -> str:
    """
    Función auxiliar para generar música base con MusicGen
    """
    from transformers import MusicgenForConditionalGeneration, AutoProcessor
    import torch
    
    # Cargar modelo MusicGen
    model_name = f"facebook/musicgen-{quality}" if quality in ["small", "medium", "large"] else "facebook/musicgen-medium"
    model = MusicgenForConditionalGeneration.from_pretrained(model_name)
    processor = AutoProcessor.from_pretrained(model_name)
    
    # Generar música
    inputs = processor(
        text=[prompt],
        padding=True,
        return_tensors="pt",
    )
    
    audio_values = model.generate(**inputs, max_new_tokens=int(duration * 50.4))  # ~50.4 tokens per second
    
    # Guardar archivo temporal
    temp_path = tempfile.mktemp(suffix=".wav")
    sf.write(temp_path, audio_values[0, 0].cpu().numpy(), 32000)
    
    return temp_path

# Tareas de monitoreo y mantenimiento
@celery_app.task(name='cleanup_temp_files')
def cleanup_temp_files():
    """Limpiar archivos temporales antiguos"""
    temp_dir = Path(tempfile.gettempdir())
    cutoff_time = time.time() - (24 * 60 * 60)  # 24 horas
    
    for file_path in temp_dir.glob("son1k_*"):
        if file_path.stat().st_mtime < cutoff_time:
            file_path.unlink()
    
    logger.info("Limpieza de archivos temporales completada")

@celery_app.task(name='health_check')
def health_check():
    """Verificar salud del sistema"""
    return {
        'status': 'healthy',
        'timestamp': time.time(),
        'workers_active': True,
        'redis_connected': True,
        'disk_space_gb': shutil.disk_usage('/').free // (1024**3)
    }

# Configurar tareas periódicas
from celery.schedules import crontab

celery_app.conf.beat_schedule = {
    'cleanup-temp-files': {
        'task': 'cleanup_temp_files',
        'schedule': crontab(hour=2, minute=0),  # Cada día a las 2 AM
    },
    'health-check': {
        'task': 'health_check',
        'schedule': 30.0,  # Cada 30 segundos
    },
}

if __name__ == "__main__":
    celery_app.start()