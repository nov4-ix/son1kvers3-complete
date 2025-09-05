"""
Son1kVers3 - Plataforma de Creación Musical con IA
Resistencia Sonora contra la homogenización musical

Este módulo contiene toda la funcionalidad core de Son1kVers3:
- Generación musical con IA (MusicGen)
- Análisis y postproducción de audio
- Clonación de voz con expresión emocional
- Ghost Studio para rearreglos creativos
- Sistema distribuido con Celery
- API REST con FastAPI
"""

__version__ = "3.0.0"
__author__ = "Son1kVers3 Team"
__email__ = "contact@son1kvers3.com"
__description__ = "Democratizando la creación musical con IA - Resistencia Sonora"

# Importaciones principales
from .main import app
from .config import settings
from .models import *
from .database import get_db
from .audio_processing import ResistanceAudioProcessor
from .voice_expression import VocalExpressionEngine
from .celery_worker import celery_app

# Configurar logging
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('son1kvers3.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
logger.info(f"🎵 Son1kVers3 v{__version__} - Resistencia Sonora Iniciada")

# Metadatos del proyecto
PROJECT_INFO = {
    "name": "Son1kVers3",
    "version": __version__,
    "description": __description__,
    "slogan": "Maqueta → Production",
    "mission": "Democratizar la creación musical",
    "narrative": "Resistencia Sonora contra la homogenización",
    "target_market": "400+ millones de hispanohablantes",
    "core_features": [
        "Generación musical con IA contextual",
        "Postproducción automática SSL/Neve",
        "Clonación de voz expresiva",
        "Ghost Studio creativo",
        "Interface A/B comparison",
        "Sistema distribuido escalable"
    ]
}

# Validar configuración en importación
try:
    from .config import settings
    logger.info("✅ Configuración cargada correctamente")
except Exception as e:
    logger.error(f"❌ Error cargando configuración: {e}")

# Verificar dependencias críticas
try:
    import torch
    import transformers
    import librosa
    import soundfile
    logger.info("✅ Dependencias de IA musical verificadas")
except ImportError as e:
    logger.warning(f"⚠️  Dependencia faltante: {e}")

__all__ = [
    'app',
    'settings', 
    'get_db',
    'ResistanceAudioProcessor',
    'VocalExpressionEngine',
    'celery_app',
    'PROJECT_INFO',
    '__version__'
]