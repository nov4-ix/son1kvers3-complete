"""
File utilities for Son1k v3.0
Handles file operations, directory setup, and cleanup
"""

import os
import shutil
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import soundfile as sf
import numpy as np

logger = logging.getLogger(__name__)

def setup_directories():
    """Create all required directories for Son1k"""
    from src.core.config import get_settings
    settings = get_settings()
    
    directories = [
        settings.storage_paths["root"],
        settings.storage_paths["uploads"],
        settings.storage_paths["output"],
        settings.storage_paths["models"],
        settings.storage_paths["ghost_uploads"],
        settings.storage_paths["ghost_output"],
        Path("backend/data"),  # For presets and jobs
        Path("logs"),          # For log files
    ]
    
    for directory in directories:
        try:
            directory.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Directory ensured: {directory}")
        except Exception as e:
            logger.error(f"Failed to create directory {directory}: {e}")
            raise
    
    # Create .gitkeep files to ensure directories are tracked by git
    gitkeep_dirs = [
        settings.storage_paths["uploads"],
        settings.storage_paths["output"],
        settings.storage_paths["models"],
    ]
    
    for directory in gitkeep_dirs:
        gitkeep_file = directory / ".gitkeep"
        if not gitkeep_file.exists():
            try:
                gitkeep_file.touch()
            except Exception as e:
                logger.warning(f"Failed to create .gitkeep in {directory}: {e}")

def save_audio_file(audio: np.ndarray, 
                   output_path: Union[str, Path],
                   sample_rate: int = 32000,
                   metadata: Optional[Dict[str, Any]] = None) -> str:
    """
    Save audio array to file with metadata
    
    Args:
        audio: Audio array to save
        output_path: Path where to save the file
        sample_rate: Audio sample rate
        metadata: Optional metadata to include
        
    Returns:
        String path of saved file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Ensure audio is in correct format
        if audio.ndim > 1:
            audio = audio.squeeze()  # Remove extra dimensions
        
        # Normalize if needed
        if np.max(np.abs(audio)) > 1.0:
            audio = audio / np.max(np.abs(audio)) * 0.95
            logger.warning("Audio was clipping, normalized to prevent distortion")
        
        # Save audio file
        sf.write(output_path, audio, sample_rate)
        
        # Save metadata if provided
        if metadata:
            metadata_path = output_path.with_suffix('.json')
            import json
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            logger.debug(f"Metadata saved: {metadata_path}")
        
        logger.debug(f"Audio saved: {output_path} ({len(audio)/sample_rate:.1f}s)")
        return str(output_path)
        
    except Exception as e:
        logger.error(f"Failed to save audio to {output_path}: {e}")
        raise

def load_audio_file(file_path: Union[str, Path], 
                   target_sample_rate: Optional[int] = None) -> tuple[np.ndarray, int]:
    """
    Load audio file and optionally resample
    
    Args:
        file_path: Path to audio file
        target_sample_rate: If provided, resample to this rate
        
    Returns:
        Tuple of (audio_array, sample_rate)
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    
    try:
        # Load audio
        audio, sample_rate = sf.read(file_path)
        
        # Convert to mono if stereo
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        
        # Resample if needed
        if target_sample_rate and sample_rate != target_sample_rate:
            import librosa
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=target_sample_rate)
            sample_rate = target_sample_rate
        
        logger.debug(f"Audio loaded: {file_path} ({len(audio)/sample_rate:.1f}s @ {sample_rate}Hz)")
        return audio, sample_rate
        
    except Exception as e:
        logger.error(f"Failed to load audio from {file_path}: {e}")
        raise

def validate_audio_file(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Validate audio file and return information
    
    Returns:
        Dict with validation results and file info
    """
    file_path = Path(file_path)
    
    result = {
        "valid": False,
        "error": None,
        "file_info": {}
    }
    
    try:
        # Check if file exists
        if not file_path.exists():
            result["error"] = "File does not exist"
            return result
        
        # Check file size
        file_size = file_path.stat().st_size
        max_size = 100 * 1024 * 1024  # 100MB
        
        if file_size > max_size:
            result["error"] = f"File too large: {file_size / 1024 / 1024:.1f}MB (max 100MB)"
            return result
        
        # Check file extension
        valid_extensions = ['.wav', '.mp3', '.flac', '.aiff', '.m4a']
        if file_path.suffix.lower() not in valid_extensions:
            result["error"] = f"Unsupported format: {file_path.suffix}"
            return result
        
        # Try to read audio info
        info = sf.info(file_path)
        
        # Check duration
        if info.duration > 60:  # 60 seconds max
            result["error"] = f"Audio too long: {info.duration:.1f}s (max 60s)"
            return result
        
        if info.duration < 0.5:  # 0.5 seconds min
            result["error"] = f"Audio too short: {info.duration:.1f}s (min 0.5s)"
            return result
        
        # File is valid
        result["valid"] = True
        result["file_info"] = {
            "duration_s": info.duration,
            "sample_rate": info.samplerate,
            "channels": info.channels,
            "format": info.format,
            "subtype": info.subtype,
            "file_size_mb": file_size / 1024 / 1024
        }
        
        return result
        
    except Exception as e:
        result["error"] = f"File validation failed: {str(e)}"
        return result

def cleanup_old_sessions(uploads_dir: Path, max_age_hours: int = 24):
    """
    Clean up old session directories
    
    Args:
        uploads_dir: Directory containing session folders
        max_age_hours: Maximum age in hours before deletion
    """
    if not uploads_dir.exists():
        return
    
    try:
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        cleaned_count = 0
        
        for session_dir in uploads_dir.iterdir():
            if not session_dir.is_dir():
                continue
            
            try:
                # Check directory age
                dir_age = current_time - session_dir.stat().st_mtime
                
                if dir_age > max_age_seconds:
                    shutil.rmtree(session_dir)
                    cleaned_count += 1
                    logger.debug(f"Cleaned up old session: {session_dir.name}")
                    
            except Exception as e:
                logger.warning(f"Failed to clean up session {session_dir}: {e}")
        
        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} old sessions")
            
    except Exception as e:
        logger.error(f"Session cleanup failed: {e}")

def cleanup_old_files(output_dir: Path, max_age_hours: int = 24, pattern: str = "*.wav"):
    """
    Clean up old generated files
    
    Args:
        output_dir: Directory containing generated files
        max_age_hours: Maximum age in hours before deletion
        pattern: File pattern to match
    """
    if not output_dir.exists():
        return
    
    try:
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        cleaned_count = 0
        
        for file_path in output_dir.glob(pattern):
            try:
                # Check file age
                file_age = current_time - file_path.stat().st_mtime
                
                if file_age > max_age_seconds:
                    file_path.unlink()
                    cleaned_count += 1
                    
                    # Also remove associated metadata file
                    metadata_file = file_path.with_suffix('.json')
                    if metadata_file.exists():
                        metadata_file.unlink()
                        
            except Exception as e:
                logger.warning(f"Failed to clean up file {file_path}: {e}")
        
        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} old files")
            
    except Exception as e:
        logger.error(f"File cleanup failed: {e}")

def get_directory_size(directory: Path) -> Dict[str, Any]:
    """
    Get directory size and file count
    
    Returns:
        Dict with size information
    """
    if not directory.exists():
        return {"size_mb": 0, "file_count": 0, "subdirs": 0}
    
    try:
        total_size = 0
        file_count = 0
        subdir_count = 0
        
        for item in directory.rglob("*"):
            if item.is_file():
                total_size += item.stat().st_size
                file_count += 1
            elif item.is_dir():
                subdir_count += 1
        
        return {
            "size_mb": total_size / 1024 / 1024,
            "file_count": file_count,
            "subdirs": subdir_count
        }
        
    except Exception as e:
        logger.error(f"Failed to get directory size for {directory}: {e}")
        return {"size_mb": 0, "file_count": 0, "subdirs": 0}

def get_storage_stats() -> Dict[str, Any]:
    """Get comprehensive storage statistics"""
    from src.core.config import get_settings
    settings = get_settings()
    
    stats = {}
    
    # Check each storage directory
    for name, path in settings.storage_paths.items():
        stats[name] = get_directory_size(path)
    
    # Total across all directories
    total_size = sum(stat["size_mb"] for stat in stats.values())
    total_files = sum(stat["file_count"] for stat in stats.values())
    
    stats["total"] = {
        "size_mb": total_size,
        "file_count": total_files
    }
    
    return stats

def create_backup(source_dir: Path, backup_dir: Path, 
                 include_patterns: List[str] = None,
                 exclude_patterns: List[str] = None) -> str:
    """
    Create backup of directory with optional filtering
    
    Args:
        source_dir: Source directory to backup
        backup_dir: Directory where backup will be created
        include_patterns: List of patterns to include (e.g., ['*.wav', '*.json'])
        exclude_patterns: List of patterns to exclude (e.g., ['*.tmp'])
        
    Returns:
        Path to created backup
    """
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")
    
    # Create backup filename with timestamp
    timestamp = int(time.time())
    backup_name = f"{source_dir.name}_backup_{timestamp}"
    backup_path = backup_dir / backup_name
    
    try:
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy directory
        shutil.copytree(source_dir, backup_path)
        
        # Apply filtering if specified
        if exclude_patterns:
            for pattern in exclude_patterns:
                for file_path in backup_path.rglob(pattern):
                    try:
                        if file_path.is_file():
                            file_path.unlink()
                        elif file_path.is_dir():
                            shutil.rmtree(file_path)
                    except Exception as e:
                        logger.warning(f"Failed to exclude {file_path}: {e}")
        
        # Create backup info file
        backup_info = {
            "created_at": timestamp,
            "source_dir": str(source_dir),
            "backup_size_mb": get_directory_size(backup_path)["size_mb"],
            "include_patterns": include_patterns,
            "exclude_patterns": exclude_patterns
        }
        
        info_file = backup_path / "backup_info.json"
        import json
        with open(info_file, 'w') as f:
            json.dump(backup_info, f, indent=2)
        
        logger.info(f"Backup created: {backup_path}")
        return str(backup_path)
        
    except Exception as e:
        logger.error(f"Backup creation failed: {e}")
        # Cleanup partial backup
        if backup_path.exists():
            shutil.rmtree(backup_path)
        raise

def safe_filename(filename: str, max_length: int = 100) -> str:
    """
    Create a safe filename by removing/replacing invalid characters
    
    Args:
        filename: Original filename
        max_length: Maximum filename length
        
    Returns:
        Safe filename string
    """
    import re
    
    # Remove or replace invalid characters
    safe_name = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Remove multiple consecutive underscores
    safe_name = re.sub(r'_+', '_', safe_name)
    
    # Remove leading/trailing whitespace and underscores
    safe_name = safe_name.strip(' _.')
    
    # Ensure it's not empty
    if not safe_name:
        safe_name = "unnamed"
    
    # Truncate if too long
    if len(safe_name) > max_length:
        name_part = safe_name[:max_length-10]
        safe_name = f"{name_part}..."
    
    return safe_name

def get_file_hash(file_path: Path) -> str:
    """Get SHA-256 hash of file for deduplication"""
    import hashlib
    
    hash_sha256 = hashlib.sha256()
    
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    except Exception as e:
        logger.error(f"Failed to hash file {file_path}: {e}")
        return ""

def deduplicate_files(directory: Path, dry_run: bool = True) -> Dict[str, Any]:
    """
    Find and optionally remove duplicate files in directory
    
    Args:
        directory: Directory to scan for duplicates
        dry_run: If True, only report duplicates without deleting
        
    Returns:
        Dict with deduplication results
    """
    if not directory.exists():
        return {"error": "Directory not found"}
    
    # Build hash map
    hash_map = {}
    duplicates = []
    
    for file_path in directory.rglob("*"):
        if file_path.is_file():
            file_hash = get_file_hash(file_path)
            if file_hash:
                if file_hash in hash_map:
                    duplicates.append({
                        "original": str(hash_map[file_hash]),
                        "duplicate": str(file_path),
                        "size_mb": file_path.stat().st_size / 1024 / 1024
                    })
                else:
                    hash_map[file_hash] = file_path
    
    # Remove duplicates if not dry run
    removed_count = 0
    saved_space_mb = 0
    
    if not dry_run:
        for dup in duplicates:
            try:
                dup_path = Path(dup["duplicate"])
                saved_space_mb += dup["size_mb"]
                dup_path.unlink()
                removed_count += 1
            except Exception as e:
                logger.warning(f"Failed to remove duplicate {dup['duplicate']}: {e}")
    
    return {
        "duplicates_found": len(duplicates),
        "duplicates_removed": removed_count,
        "space_saved_mb": saved_space_mb,
        "duplicates": duplicates[:10]  # Limit output
    }