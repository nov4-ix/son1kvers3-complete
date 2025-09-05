"""
Ghost Studio API Endpoints for Son1k v3.0
Handles automated music generation, presets, and Maqueta → Production workflow
"""

from fastapi import APIRouter, HTTPException, Depends, File, UploadFile, Form, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
import uuid
import time
import json
import asyncio
from pathlib import Path
import logging
import shutil
import soundfile as sf
import numpy as np

# Local imports
from src.config import settings
from src.services.musicgen import MusicGenService
from src.services.audio_analysis import AudioAnalyzer
from src.services.audio_post import AudioPostProcessor
from src.utils.file_utils import save_audio_file, cleanup_old_sessions

logger = logging.getLogger(__name__)
router = APIRouter()

# Models
class JobCreate(BaseModel):
    preset: str
    prompt_extra: Optional[str] = ""
    duration: Optional[float] = None

class MaquetaProcessRequest(BaseModel):
    prompt: str = Field(..., description="Vision for the production")
    duration: float = Field(12.0, description="Duration in seconds")
    tune_amount: float = Field(0.7, description="Pitch correction strength")
    eq_low_gain: float = Field(1.5, description="EQ low gain in dB")
    eq_mid1_gain: float = Field(-1.0, description="EQ mid1 gain in dB")
    eq_mid2_gain: float = Field(1.5, description="EQ mid2 gain in dB")
    eq_high_gain: float = Field(1.0, description="EQ high gain in dB")
    sat_drive: float = Field(6.0, description="Saturation drive in dB")
    sat_mix: float = Field(0.35, description="Saturation mix")
    lufs_target: float = Field(-14.0, description="LUFS target level")

# Global storage
# settings ya importado
_jobs_storage = {}
_sessions_storage = {}

# Load presets
def load_presets() -> Dict[str, Any]:
    """Load Ghost Studio presets from JSON file"""
    presets_file = Path(settings.GHOST_PRESETS_FILE)
    
    if not presets_file.exists():
        # Create default presets
        default_presets = {
            "latin_rock": {
                "name": "Latin Rock",
                "description": "Energetic Latin rock with guitars and percussion",
                "prompt_base": "latin rock with electric guitars, congas, timbales, energetic drums, 120 bpm, major key",
                "suggested_bpm": 120,
                "suggested_duration": 12,
                "seed": 42,
                "tags": ["rock", "latin", "energetic", "guitar"],
                "parameters": {
                    "temperature": 1.0,
                    "top_k": 250,
                    "top_p": 0.9
                }
            },
            "trap_808": {
                "name": "Trap 808",
                "description": "Heavy trap beat with 808 drums and dark atmosphere",
                "prompt_base": "trap beat with heavy 808 drums, dark synths, hi-hats, 140 bpm, minor key, urban",
                "suggested_bpm": 140,
                "suggested_duration": 15,
                "seed": 808,
                "tags": ["trap", "hip-hop", "808", "dark", "urban"],
                "parameters": {
                    "temperature": 1.1,
                    "top_k": 200,
                    "top_p": 0.85
                }
            },
            "ambient_cinematique": {
                "name": "Ambient Cinematic",
                "description": "Atmospheric cinematic soundscape with orchestral elements",
                "prompt_base": "ambient cinematic music with strings, soft piano, atmospheric pads, 70 bpm, ethereal",
                "suggested_bpm": 70,
                "suggested_duration": 20,
                "seed": 2001,
                "tags": ["ambient", "cinematic", "orchestral", "atmospheric", "emotional"],
                "parameters": {
                    "temperature": 0.9,
                    "top_k": 300,
                    "top_p": 0.95
                }
            },
            "synthwave_retro": {
                "name": "Synthwave Retro",
                "description": "80s inspired synthwave with retro aesthetics",
                "prompt_base": "synthwave with retro synthesizers, drum machine, 80s style, 110 bpm, nostalgic",
                "suggested_bpm": 110,
                "suggested_duration": 18,
                "seed": 1984,
                "tags": ["synthwave", "retro", "80s", "electronic", "cyberpunk"],
                "parameters": {
                    "temperature": 1.0,
                    "top_k": 250,
                    "top_p": 0.9
                }
            },
            "jazz_fusion": {
                "name": "Jazz Fusion",
                "description": "Complex jazz fusion with electric instruments",
                "prompt_base": "jazz fusion with electric guitar, electric piano, bass, complex rhythms, 130 bpm",
                "suggested_bpm": 130,
                "suggested_duration": 16,
                "seed": 1959,
                "tags": ["jazz", "fusion", "complex", "electric", "sophisticated"],
                "parameters": {
                    "temperature": 1.2,
                    "top_k": 150,
                    "top_p": 0.8
                }
            }
        }
        
        # Save default presets
        presets_file.parent.mkdir(parents=True, exist_ok=True)
        with open(presets_file, 'w') as f:
            json.dump(default_presets, f, indent=2)
        
        return default_presets
    
    try:
        with open(presets_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load presets: {e}")
        raise HTTPException(status_code=500, detail="Failed to load presets")

@router.get("/presets")
async def get_presets():
    """Get all available Ghost Studio presets"""
    try:
        presets = load_presets()
        return {
            "presets": presets,
            "count": len(presets)
        }
    except Exception as e:
        logger.error(f"Failed to get presets: {e}")
        raise HTTPException(status_code=500, detail="Failed to load presets")

@router.post("/job")
async def create_ghost_job(job_data: JobCreate, 
                          background_tasks: BackgroundTasks,
                          musicgen_service: MusicGenService = Depends()):
    """Create a new Ghost Studio job"""
    try:
        # Load presets
        presets = load_presets()
        
        if job_data.preset not in presets:
            raise HTTPException(status_code=400, detail=f"Preset '{job_data.preset}' not found")
        
        preset = presets[job_data.preset]
        
        # Generate job ID
        job_id = str(uuid.uuid4())
        
        # Prepare job
        job = {
            "id": job_id,
            "preset": job_data.preset,
            "prompt_extra": job_data.prompt_extra or "",
            "duration": job_data.duration or preset["suggested_duration"],
            "status": "queued",
            "created_at": time.time(),
            "started_at": None,
            "completed_at": None,
            "output_url": None,
            "error_message": None
        }
        
        # Store job
        _jobs_storage[job_id] = job
        
        # Start job in background
        background_tasks.add_task(process_ghost_job, job_id, preset, musicgen_service)
        
        logger.info(f"Ghost job created: {job_id} ({job_data.preset})")
        
        return {
            "ok": True,
            "job_id": job_id,
            "status": "queued",
            "message": "Job created and processing started"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create job: {e}")
        raise HTTPException(status_code=500, detail=f"Job creation failed: {str(e)}")

async def process_ghost_job(job_id: str, preset: Dict[str, Any], musicgen_service: MusicGenService):
    """Process a Ghost Studio job in background"""
    try:
        # Update job status
        job = _jobs_storage[job_id]
        job["status"] = "running"
        job["started_at"] = time.time()
        
        logger.info(f"Processing Ghost job: {job_id}")
        
        # Ensure model is loaded
        if not musicgen_service.is_loaded():
            await musicgen_service.load_model()
        
        # Build prompt
        base_prompt = preset["prompt_base"]
        extra_prompt = job["prompt_extra"]
        
        if extra_prompt:
            final_prompt = f"{base_prompt}, {extra_prompt}"
        else:
            final_prompt = base_prompt
        
        # Generate music
        audio, metadata = musicgen_service.generate_music(
            prompt=final_prompt,
            duration=job["duration"],
            seed=preset.get("seed"),
            **preset.get("parameters", {})
        )
        
        # Save audio
        timestamp = int(time.time())
        filename = f"ghost_{job_id[:8]}_{timestamp}.wav"
        output_path = settings.storage_paths["output"] / filename
        
        musicgen_service.save_audio(audio, output_path, metadata)
        
        # Update job
        job["status"] = "done"
        job["completed_at"] = time.time()
        job["output_url"] = f"/output/{filename}"
        job["metadata"] = metadata
        
        logger.info(f"Ghost job completed: {job_id}")
        
    except Exception as e:
        logger.error(f"Ghost job failed: {job_id} - {e}")
        
        # Update job with error
        job = _jobs_storage.get(job_id, {})
        job["status"] = "error"
        job["error_message"] = str(e)
        job["completed_at"] = time.time()

@router.get("/jobs")
async def get_ghost_jobs(limit: int = 50, status: Optional[str] = None):
    """Get Ghost Studio jobs with optional filtering"""
    try:
        jobs = list(_jobs_storage.values())
        
        # Filter by status if provided
        if status:
            jobs = [job for job in jobs if job.get("status") == status]
        
        # Sort by creation time (newest first)
        jobs.sort(key=lambda x: x.get("created_at", 0), reverse=True)
        
        # Limit results
        jobs = jobs[:limit]
        
        return {
            "jobs": jobs,
            "count": len(jobs),
            "total_count": len(_jobs_storage)
        }
        
    except Exception as e:
        logger.error(f"Failed to get jobs: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve jobs")

@router.get("/jobs/{job_id}")
async def get_ghost_job(job_id: str):
    """Get specific Ghost Studio job status"""
    if job_id not in _jobs_storage:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = _jobs_storage[job_id]
    
    return {
        "id": job["id"],
        "status": job["status"],
        "output_url": job.get("output_url"),
        "error_message": job.get("error_message"),
        "progress": f"Job {job['status']}"
    }

@router.delete("/jobs/{job_id}")
async def delete_ghost_job(job_id: str):
    """Delete a Ghost Studio job"""
    if job_id not in _jobs_storage:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = _jobs_storage[job_id]
    
    # Don't allow deletion of running jobs
    if job.get("status") == "running":
        raise HTTPException(status_code=400, detail="Cannot delete running job")
    
    # Remove output file if exists
    if job.get("output_url"):
        try:
            output_path = settings.storage_paths["output"] / job["output_url"].split("/")[-1]
            if output_path.exists():
                output_path.unlink()
        except Exception as e:
            logger.warning(f"Failed to delete output file: {e}")
    
    # Remove job
    del _jobs_storage[job_id]
    
    return {
        "ok": True,
        "message": f"Job {job_id[:8]}... deleted"
    }

@router.get("/stats")
async def get_ghost_stats():
    """Get Ghost Studio statistics"""
    try:
        # Job statistics
        total_jobs = len(_jobs_storage)
        status_breakdown = {}
        
        for job in _jobs_storage.values():
            status = job.get("status", "unknown")
            status_breakdown[status] = status_breakdown.get(status, 0) + 1
        
        # Storage statistics
        upload_dir = settings.storage_paths["ghost_uploads"]
        output_dir = settings.storage_paths["ghost_output"]
        
        storage_stats = {
            "sessions": len(list(upload_dir.glob("*"))) if upload_dir.exists() else 0,
            "productions": len(list(output_dir.glob("*.wav"))) if output_dir.exists() else 0
        }
        
        # Preset statistics
        presets = load_presets()
        
        return {
            "sessions": len(_sessions_storage),
            "total_jobs": total_jobs,
            "status_breakdown": status_breakdown,
            "storage": storage_stats,
            "presets": len(presets),
            "available_presets": list(presets.keys())
        }
        
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve statistics")

@router.post("/maqueta")
async def process_maqueta_to_production(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    prompt: str = Form(...),
    duration: float = Form(12.0),
    tune_amount: float = Form(0.7),
    eq_low_gain: float = Form(1.5),
    eq_mid1_gain: float = Form(-1.0),
    eq_mid2_gain: float = Form(1.5),
    eq_high_gain: float = Form(1.0),
    sat_drive: float = Form(6.0),
    sat_mix: float = Form(0.35),
    lufs_target: float = Form(-14.0),
    musicgen_service: MusicGenService = Depends()
):
    """
    Complete Maqueta → Production workflow
    Upload demo, analyze, generate AI production with professional postprocessing
    """
    session_id = str(uuid.uuid4())
    start_time = time.time()
    
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Validate file format and size
        valid, error_msg = settings.validate_file_upload(file.filename, 0)  # Size check done by FastAPI
        if not valid:
            raise HTTPException(status_code=400, detail=error_msg)
        
        logger.info(f"Starting maqueta processing: {session_id}")
        
        # Create session directory
        session_dir = settings.storage_paths["ghost_uploads"] / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        
        # Save uploaded file
        demo_path = session_dir / f"demo{Path(file.filename).suffix}"
        with open(demo_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Initialize services
        analyzer = AudioAnalyzer(sample_rate=settings.SAMPLE_RATE)
        postprocessor = AudioPostProcessor(sample_rate=settings.SAMPLE_RATE)
        
        # 1. Analyze demo
        logger.info(f"Analyzing demo: {session_id}")
        analysis = analyzer.analyze_audio_file(str(demo_path))
        
        # 2. Build AI generation prompt
        prompt_final = build_generation_prompt(analysis, prompt)
        
        # 3. Ensure MusicGen model is loaded
        if not musicgen_service.is_loaded():
            await musicgen_service.load_model()
        
        # 4. Generate AI music
        logger.info(f"Generating AI production: {session_id}")
        generated_audio, gen_metadata = musicgen_service.generate_music(
            prompt=prompt_final,
            duration=duration,
            temperature=1.0,
            top_k=250
        )
        
        # 5. Professional postprocessing
        logger.info(f"Applying professional postprocessing: {session_id}")
        
        # Check if vocals detected for tuning
        has_vocals = analysis.get("vocals", {}).get("has_vocals", False)
        key_root = analysis.get("key_guess", {}).get("root", "C")
        key_scale = analysis.get("key_guess", {}).get("scale", "major")
        
        processed_audio, post_metadata = postprocessor.process_master(
            generated_audio,
            eq_params={
                "low_gain_db": eq_low_gain,
                "mid1_gain_db": eq_mid1_gain,
                "mid2_gain_db": eq_mid2_gain,
                "high_gain_db": eq_high_gain
            },
            tune_params={
                "enabled": has_vocals and tune_amount > 0.1,
                "key_root": key_root,
                "key_scale": key_scale,
                "strength": tune_amount
            },
            sat_params={
                "drive_db": sat_drive,
                "mix": sat_mix
            },
            master_params={
                "lufs_target": lufs_target,
                "ceiling_db": -0.3,
                "fade_in_ms": 50,
                "fade_out_ms": 200
            }
        )
        
        # 6. Save production
        production_dir = settings.storage_paths["ghost_output"] / session_id
        production_dir.mkdir(parents=True, exist_ok=True)
        
        production_path = production_dir / "production.wav"
        musicgen_service.save_audio(processed_audio, production_path)
        
        # 7. Store session
        processing_time = time.time() - start_time
        
        session_data = {
            "session_id": session_id,
            "created_at": start_time,
            "processing_time_s": processing_time,
            "prompt_original": prompt,
            "prompt_final": prompt_final,
            "demo": {
                "filename": demo_path.name,
                "url": f"/uploads/ghost/{session_id}/{demo_path.name}",
                "analysis": analysis,
                "duration_s": analysis["file_info"]["duration_s"]
            },
            "production": {
                "filename": "production.wav",
                "url": f"/output/ghost/{session_id}/production.wav",
                "duration_s": len(processed_audio) / settings.SAMPLE_RATE,
                "device": gen_metadata["device"],
                "post_metadata": post_metadata
            },
            "parameters": {
                "duration": duration,
                "tune_amount": tune_amount,
                "eq_params": {
                    "low_gain": eq_low_gain,
                    "mid1_gain": eq_mid1_gain,
                    "mid2_gain": eq_mid2_gain,
                    "high_gain": eq_high_gain
                },
                "sat_params": {"drive": sat_drive, "mix": sat_mix},
                "lufs_target": lufs_target
            }
        }
        
        _sessions_storage[session_id] = session_data
        
        # Schedule cleanup of old sessions
        background_tasks.add_task(cleanup_old_sessions, settings.storage_paths["ghost_uploads"])
        
        logger.info(f"Maqueta processing complete: {session_id} ({processing_time:.1f}s)")
        
        return {
            "ok": True,
            **session_data
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Maqueta processing failed: {session_id} - {e}")
        
        # Cleanup on error
        try:
            session_dir = settings.storage_paths["ghost_uploads"] / session_id
            if session_dir.exists():
                shutil.rmtree(session_dir)
            production_dir = settings.storage_paths["ghost_output"] / session_id
            if production_dir.exists():
                shutil.rmtree(production_dir)
        except:
            pass
        
        raise HTTPException(status_code=500, detail=f"Maqueta processing failed: {str(e)}")

def build_generation_prompt(analysis: Dict[str, Any], user_prompt: str) -> str:
    """Build intelligent AI generation prompt from analysis + user vision"""
    try:
        # Extract key characteristics
        tempo = analysis.get("tempo", {})
        key_info = analysis.get("key_guess", {})
        vocals = analysis.get("vocals", {})
        summary = analysis.get("summary", {})
        
        # Build context
        context_parts = []
        
        # Tempo information
        if tempo.get("bpm"):
            bpm = tempo["bpm"]
            context_parts.append(f"tempo around {bpm:.0f} BPM")
        
        # Key information
        if key_info.get("root") and key_info.get("scale"):
            key_sig = f"{key_info['root']} {key_info['scale']}"
            context_parts.append(f"key of {key_sig}")
        
        # Energy level
        energy_level = summary.get("energy_level", "medium")
        if energy_level != "medium":
            context_parts.append(f"{energy_level} energy")
        
        # Vocal presence
        if vocals.get("has_vocals"):
            vocal_prob = vocals.get("vocal_probability", 0)
            if vocal_prob > 0.7:
                context_parts.append("includes vocal elements")
            else:
                context_parts.append("includes possible vocal elements")
        
        # Characteristics
        characteristics = summary.get("characteristics", [])
        if characteristics:
            char_str = ", ".join(characteristics[:2])  # Limit to 2 characteristics
            context_parts.append(char_str)
        
        # Build final prompt
        if context_parts:
            context = "Using detected musical characteristics: " + ", ".join(context_parts)
            final_prompt = f"{context}. Transform this musical foundation into: {user_prompt}"
        else:
            final_prompt = f"Create music based on this vision: {user_prompt}"
        
        # Ensure reasonable length
        if len(final_prompt) > 250:
            final_prompt = final_prompt[:250].rsplit(' ', 1)[0] + "..."
        
        return final_prompt
        
    except Exception as e:
        logger.warning(f"Prompt building failed: {e}")
        return f"Create music: {user_prompt}"

@router.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """Get maqueta session details"""
    if session_id not in _sessions_storage:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return _sessions_storage[session_id]

@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a maqueta session and its files"""
    if session_id not in _sessions_storage:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        # Remove files
        session_dir = settings.storage_paths["ghost_uploads"] / session_id
        if session_dir.exists():
            shutil.rmtree(session_dir)
        
        production_dir = settings.storage_paths["ghost_output"] / session_id
        if production_dir.exists():
            shutil.rmtree(production_dir)
        
        # Remove from storage
        del _sessions_storage[session_id]
        
        return {
            "ok": True,
            "message": f"Session {session_id[:8]}... deleted"
        }
        
    except Exception as e:
        logger.error(f"Session deletion failed: {e}")
        raise HTTPException(status_code=500, detail="Session deletion failed")