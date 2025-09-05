"""
Manual Music Generation API for Son1k v3.0
Handles direct music generation with full parameter control
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
import time
import logging
from pathlib import Path

# Local imports
from src.core.config import get_settings
from src.services.musicgen import MusicGenService, validate_prompt, estimate_generation_time
from src.services.audio_post import AudioPostProcessor

logger = logging.getLogger(__name__)
router = APIRouter()

# Request models
class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Text description of the music to generate")
    duration: float = Field(8.0, ge=1.0, le=30.0, description="Duration in seconds")
    temperature: float = Field(1.0, ge=0.1, le=2.0, description="Sampling temperature")
    top_k: int = Field(250, ge=1, le=1000, description="Top-k sampling parameter")
    top_p: float = Field(0.0, ge=0.0, le=1.0, description="Top-p (nucleus) sampling")
    seed: Optional[int] = Field(None, description="Random seed for reproducible generation")
    apply_postprocessing: bool = Field(True, description="Apply basic postprocessing")

class GenerateResponse(BaseModel):
    ok: bool
    url: str
    filename: str
    prompt: str
    duration: float
    generation_time: float
    model: str
    device: str
    parameters: Dict[str, Any]
    postprocessing_applied: bool
    audio_stats: Dict[str, float]

# Global settings
settings = get_settings()

@router.post("/generate", response_model=GenerateResponse)
async def generate_music(
    request: GenerateRequest,
    background_tasks: BackgroundTasks,
    musicgen_service: MusicGenService = Depends()
):
    """
    Generate music from text prompt with full parameter control
    """
    try:
        logger.info(f"Manual generation request: '{request.prompt}' ({request.duration}s)")
        start_time = time.time()
        
        # Validate prompt
        validated_prompt = validate_prompt(request.prompt)
        
        # Estimate generation time for user feedback
        estimated_time = estimate_generation_time(request.duration, musicgen_service.device)
        logger.debug(f"Estimated generation time: {estimated_time:.1f}s")
        
        # Ensure model is loaded
        if not musicgen_service.is_loaded():
            logger.info("Loading MusicGen model...")
            await musicgen_service.load_model()
        
        # Generate music
        audio, metadata = musicgen_service.generate_music(
            prompt=validated_prompt,
            duration=request.duration,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
            seed=request.seed
        )
        
        # Apply basic postprocessing if requested
        postprocessing_applied = False
        if request.apply_postprocessing:
            try:
                processor = AudioPostProcessor(sample_rate=settings.SAMPLE_RATE)
                
                # Basic mastering chain
                audio, post_metadata = processor.process_master(
                    audio,
                    eq_params={
                        "low_gain_db": 0.5,
                        "mid1_gain_db": 0.0,
                        "mid2_gain_db": 0.5,
                        "high_gain_db": 0.5
                    },
                    tune_params={"enabled": False},
                    sat_params={"drive_db": 2.0, "mix": 0.15},
                    master_params={
                        "lufs_target": -14.0,
                        "ceiling_db": -0.3,
                        "fade_in_ms": 50,
                        "fade_out_ms": 200
                    }
                )
                
                # Update metadata
                metadata.update(post_metadata)
                postprocessing_applied = True
                logger.debug("Basic postprocessing applied")
                
            except Exception as e:
                logger.warning(f"Postprocessing failed, using raw audio: {e}")
        
        # Generate filename
        timestamp = int(time.time())
        safe_prompt = "".join(c for c in validated_prompt[:20] if c.isalnum() or c in (' ', '-', '_')).strip()
        safe_prompt = safe_prompt.replace(' ', '_')
        filename = f"son1k_{safe_prompt}_{timestamp}.wav"
        
        # Save audio
        output_path = settings.storage_paths["output"] / filename
        musicgen_service.save_audio(audio, output_path, metadata)
        
        total_time = time.time() - start_time
        
        # Prepare response
        response_data = {
            "ok": True,
            "url": f"/output/{filename}",
            "filename": filename,
            "prompt": validated_prompt,
            "duration": metadata["duration_s"],
            "generation_time": total_time,
            "model": metadata["model"],
            "device": metadata["device"],
            "parameters": {
                "temperature": request.temperature,
                "top_k": request.top_k,
                "top_p": request.top_p,
                "seed": request.seed,
                "sample_rate": metadata["sample_rate"]
            },
            "postprocessing_applied": postprocessing_applied,
            "audio_stats": metadata["audio_stats"]
        }
        
        logger.info(f"Generation complete: {filename} ({total_time:.1f}s)")
        
        # Schedule cleanup of old files in background
        background_tasks.add_task(cleanup_old_files)
        
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Music generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@router.get("/generate/presets")
async def get_generation_presets():
    """Get predefined generation presets for quick access"""
    presets = {
        "quick_demo": {
            "name": "Quick Demo",
            "description": "Fast generation for testing",
            "prompt": "upbeat electronic music",
            "duration": 5.0,
            "temperature": 1.0,
            "top_k": 250,
            "top_p": 0.9
        },
        "rock_anthem": {
            "name": "Rock Anthem",
            "description": "Energetic rock music",
            "prompt": "epic rock anthem with electric guitars, powerful drums, 120 bpm",
            "duration": 15.0,
            "temperature": 1.1,
            "top_k": 200,
            "top_p": 0.85
        },
        "ambient_chill": {
            "name": "Ambient Chill",
            "description": "Relaxing ambient music",
            "prompt": "ambient chill music with soft synthesizers, peaceful atmosphere",
            "duration": 20.0,
            "temperature": 0.8,
            "top_k": 300,
            "top_p": 0.95
        },
        "jazz_improv": {
            "name": "Jazz Improvisation",
            "description": "Jazz with improvisation",
            "prompt": "jazz music with piano, bass, drums, improvisation, swing rhythm",
            "duration": 12.0,
            "temperature": 1.3,
            "top_k": 150,
            "top_p": 0.7
        },
        "electronic_dance": {
            "name": "Electronic Dance",
            "description": "High-energy EDM",
            "prompt": "electronic dance music with synthesizers, heavy bass, 128 bpm",
            "duration": 18.0,
            "temperature": 1.0,
            "top_k": 250,
            "top_p": 0.9
        }
    }
    
    return {
        "presets": presets,
        "count": len(presets)
    }

@router.post("/generate/batch")
async def generate_music_batch(
    requests: list[GenerateRequest],
    background_tasks: BackgroundTasks,
    musicgen_service: MusicGenService = Depends()
):
    """Generate multiple music pieces in batch"""
    if len(requests) > 5:
        raise HTTPException(status_code=400, detail="Maximum 5 requests per batch")
    
    if not musicgen_service.is_loaded():
        await musicgen_service.load_model()
    
    results = []
    total_start_time = time.time()
    
    for i, request in enumerate(requests):
        try:
            logger.info(f"Batch generation {i+1}/{len(requests)}: '{request.prompt}'")
            
            # Generate individual piece
            audio, metadata = musicgen_service.generate_music(
                prompt=validate_prompt(request.prompt),
                duration=request.duration,
                temperature=request.temperature,
                top_k=request.top_k,
                top_p=request.top_p,
                seed=request.seed
            )
            
            # Apply postprocessing if requested
            if request.apply_postprocessing:
                try:
                    processor = AudioPostProcessor(sample_rate=settings.SAMPLE_RATE)
                    audio, post_metadata = processor.process_master(audio)
                    metadata.update(post_metadata)
                except Exception as e:
                    logger.warning(f"Batch postprocessing failed for item {i+1}: {e}")
            
            # Save
            timestamp = int(time.time())
            filename = f"son1k_batch_{i+1}_{timestamp}.wav"
            output_path = settings.storage_paths["output"] / filename
            musicgen_service.save_audio(audio, output_path, metadata)
            
            results.append({
                "index": i,
                "success": True,
                "url": f"/output/{filename}",
                "filename": filename,
                "prompt": request.prompt,
                "duration": metadata["duration_s"]
            })
            
        except Exception as e:
            logger.error(f"Batch generation failed for item {i+1}: {e}")
            results.append({
                "index": i,
                "success": False,
                "error": str(e),
                "prompt": request.prompt
            })
    
    total_time = time.time() - total_start_time
    successful_count = sum(1 for r in results if r.get("success"))
    
    return {
        "ok": True,
        "results": results,
        "summary": {
            "total_requests": len(requests),
            "successful": successful_count,
            "failed": len(requests) - successful_count,
            "total_time": total_time
        }
    }

@router.get("/generate/tips")
async def get_generation_tips():
    """Get tips for better music generation prompts"""
    tips = {
        "prompt_guidelines": [
            "Be specific about genre and style (e.g., 'jazz fusion', 'ambient electronic')",
            "Include tempo information (e.g., '120 bpm', 'slow tempo', 'fast-paced')",
            "Mention key instruments (e.g., 'piano', 'electric guitar', 'synthesizer')",
            "Describe the mood or energy (e.g., 'uplifting', 'dark', 'peaceful')",
            "Add musical structure hints (e.g., 'with chorus', 'instrumental', 'solo section')"
        ],
        "good_examples": [
            "upbeat rock music with electric guitars and drums, 130 bpm, energetic",
            "ambient electronic with soft synthesizers, slow tempo, peaceful atmosphere",
            "jazz fusion with piano, bass, and complex rhythms, sophisticated",
            "latin salsa with brass section, congas, and upbeat rhythm, 100 bpm",
            "classical piano piece, romantic style, gentle and expressive"
        ],
        "parameter_tips": {
            "temperature": {
                "low": "0.5-0.8 for more predictable, structured music",
                "medium": "0.9-1.1 for balanced creativity and structure",
                "high": "1.2-2.0 for experimental, avant-garde results"
            },
            "top_k": {
                "description": "Controls vocabulary diversity",
                "conservative": "100-200 for focused generation",
                "balanced": "200-300 for good variety",
                "creative": "300-500 for maximum diversity"
            },
            "duration": {
                "short": "3-8 seconds for quick experiments",
                "medium": "8-15 seconds for complete musical ideas",
                "long": "15-30 seconds for full compositions"
            }
        },
        "common_mistakes": [
            "Prompts too vague (e.g., just 'music' or 'song')",
            "Conflicting descriptions (e.g., 'fast slow music')",
            "Too many contradictory genres in one prompt",
            "Overly complex prompts with too many details",
            "Not considering the duration vs. complexity trade-off"
        ]
    }
    
    return tips

@router.get("/generate/models")
async def get_generation_models(musicgen_service: MusicGenService = Depends()):
    """Get information about available generation models"""
    from src.services.musicgen import get_available_models
    
    models = get_available_models()
    current_model_info = musicgen_service.get_model_info()
    
    return {
        "available_models": models,
        "current_model": current_model_info,
        "device_info": {
            "device": musicgen_service.device,
            "memory_usage": current_model_info.get("memory_usage", {}),
            "model_loaded": musicgen_service.is_loaded()
        }
    }

async def cleanup_old_files():
    """Clean up old generated files to save space"""
    try:
        output_dir = settings.storage_paths["output"]
        if not output_dir.exists():
            return
        
        import os
        current_time = time.time()
        max_age_hours = 24  # Keep files for 24 hours
        max_age_seconds = max_age_hours * 3600
        
        cleaned_count = 0
        for file_path in output_dir.glob("son1k_*.wav"):
            try:
                file_age = current_time - os.path.getmtime(file_path)
                if file_age > max_age_seconds:
                    file_path.unlink()
                    cleaned_count += 1
            except Exception as e:
                logger.warning(f"Failed to clean up {file_path}: {e}")
        
        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} old generated files")
            
    except Exception as e:
        logger.error(f"File cleanup failed: {e}")