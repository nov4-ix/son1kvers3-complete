"""
MusicGen Integration Service for Son1k v3.0
Handles music generation with Facebook's MusicGen model
"""

import torch
import torchaudio
import numpy as np
from pathlib import Path
import logging
import time
from typing import Dict, Any, Optional, Union, List
import warnings
import threading
import gc

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    from transformers import MusicgenForConditionalGeneration, AutoProcessor
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.error("transformers not available - music generation disabled")

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False
    logging.warning("soundfile not available - using torchaudio for audio I/O")

logger = logging.getLogger(__name__)

class MusicGenService:
    """MusicGen service for AI music generation"""
    
    def __init__(self, model_name: str = "facebook/musicgen-small"):
        self.model_name = model_name
        self.model = None
        self.processor = None
        self.device = self._get_best_device()
        self.sample_rate = 32000
        self.max_duration = 30.0  # Max 30 seconds
        self.generation_lock = threading.Lock()
        self.model_loaded = False
        
        logger.info(f"MusicGen service initialized: {model_name} on {self.device}")
    
    def _get_best_device(self) -> str:
        """Determine the best available device"""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    async def load_model(self) -> bool:
        """Load MusicGen model asynchronously"""
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("transformers library not available")
        
        if self.model_loaded:
            logger.debug("Model already loaded")
            return True
        
        try:
            logger.info(f"Loading MusicGen model: {self.model_name}")
            start_time = time.time()
            
            # Load processor
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            
            # Load model with optimizations
            self.model = MusicgenForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                low_cpu_mem_usage=True
            )
            
            # Move to device if not using device_map
            if self.device != "cuda":
                self.model = self.model.to(self.device)
            
            # Set to evaluation mode
            self.model.eval()
            
            # Enable optimizations
            if self.device == "cuda":
                # Enable flash attention if available
                try:
                    self.model = torch.compile(self.model, mode="reduce-overhead")
                    logger.debug("Model compiled with torch.compile")
                except Exception as e:
                    logger.debug(f"Torch compile failed: {e}")
            
            load_time = time.time() - start_time
            self.model_loaded = True
            
            logger.info(f"Model loaded successfully in {load_time:.1f}s")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model = None
            self.processor = None
            self.model_loaded = False
            raise RuntimeError(f"Model loading failed: {str(e)}")
    
    def is_loaded(self) -> bool:
        """Check if model is loaded and ready"""
        return self.model_loaded and self.model is not None and self.processor is not None
    
    def generate_music(self, prompt: str, 
                      duration: float = 8.0,
                      temperature: float = 1.0,
                      top_k: int = 250,
                      top_p: float = 0.0,
                      seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Generate music from text prompt
        
        Args:
            prompt: Text description of the music
            duration: Duration in seconds (max 30)
            temperature: Sampling temperature (0.1-2.0)
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            seed: Random seed for reproducible generation
            
        Returns:
            Tuple of (audio_array, metadata)
        """
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Validate parameters
        duration = max(1.0, min(duration, self.max_duration))
        temperature = max(0.1, min(temperature, 2.0))
        top_k = max(1, min(top_k, 1000))
        top_p = max(0.0, min(top_p, 1.0))
        
        with self.generation_lock:
            try:
                logger.info(f"Generating music: '{prompt}' ({duration}s)")
                start_time = time.time()
                
                # Set random seed if provided
                if seed is not None:
                    torch.manual_seed(seed)
                    np.random.seed(seed)
                
                # Prepare inputs
                inputs = self.processor(
                    text=[prompt],
                    padding=True,
                    return_tensors="pt"
                ).to(self.device)
                
                # Calculate number of tokens for duration
                # MusicGen generates at ~50 tokens per second
                tokens_per_second = 50
                max_new_tokens = int(duration * tokens_per_second)
                
                # Generation parameters
                generation_kwargs = {
                    "max_new_tokens": max_new_tokens,
                    "do_sample": True,
                    "temperature": temperature,
                    "pad_token_id": self.processor.tokenizer.pad_token_id,
                    "eos_token_id": self.processor.tokenizer.eos_token_id
                }
                
                # Add sampling parameters
                if top_k > 0:
                    generation_kwargs["top_k"] = top_k
                if top_p > 0:
                    generation_kwargs["top_p"] = top_p
                
                # Generate with no gradient computation
                with torch.no_grad():
                    if self.device == "cuda":
                        with torch.cuda.amp.autocast():
                            generated_ids = self.model.generate(**inputs, **generation_kwargs)
                    else:
                        generated_ids = self.model.generate(**inputs, **generation_kwargs)
                
                # Decode to audio
                audio_values = self.model.generate_audio(generated_ids)
                
                # Convert to numpy
                if isinstance(audio_values, torch.Tensor):
                    audio_np = audio_values.cpu().numpy().squeeze()
                else:
                    audio_np = np.array(audio_values).squeeze()
                
                # Ensure mono
                if audio_np.ndim > 1:
                    audio_np = audio_np[0]  # Take first channel
                
                # Normalize to prevent clipping
                if np.max(np.abs(audio_np)) > 0:
                    audio_np = audio_np / np.max(np.abs(audio_np)) * 0.95
                
                generation_time = time.time() - start_time
                
                # Metadata
                metadata = {
                    "prompt": prompt,
                    "duration_s": len(audio_np) / self.sample_rate,
                    "sample_rate": self.sample_rate,
                    "generation_time_s": generation_time,
                    "model": self.model_name,
                    "device": self.device,
                    "parameters": {
                        "temperature": temperature,
                        "top_k": top_k,
                        "top_p": top_p,
                        "seed": seed,
                        "max_new_tokens": max_new_tokens
                    },
                    "audio_stats": {
                        "peak": float(np.max(np.abs(audio_np))),
                        "rms": float(np.sqrt(np.mean(audio_np ** 2))),
                        "length_samples": len(audio_np)
                    }
                }
                
                logger.info(f"Generation complete: {metadata['duration_s']:.1f}s audio in {generation_time:.1f}s")
                
                # Cleanup GPU memory
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                
                return audio_np, metadata
                
            except Exception as e:
                logger.error(f"Music generation failed: {e}")
                
                # Cleanup on error
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                
                raise RuntimeError(f"Music generation failed: {str(e)}")
    
    def generate_continuation(self, audio_prompt: np.ndarray,
                             text_prompt: str = "",
                             duration: float = 8.0,
                             **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Generate music continuation from audio prompt
        """
        if not self.is_loaded():
            raise RuntimeError("Model not loaded")
        
        # This is a simplified version - full implementation would need
        # audio conditioning capabilities
        logger.warning("Audio continuation not fully implemented, using text generation")
        
        # Fallback to text generation with enhanced prompt
        enhanced_prompt = f"continuation of audio with {text_prompt}" if text_prompt else "musical continuation"
        
        return self.generate_music(enhanced_prompt, duration, **kwargs)
    
    def save_audio(self, audio: np.ndarray, 
                   output_path: Union[str, Path],
                   metadata: Optional[Dict] = None) -> str:
        """Save generated audio to file"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            if SOUNDFILE_AVAILABLE:
                # Use soundfile for better format support
                sf.write(output_path, audio, self.sample_rate)
            else:
                # Fallback to torchaudio
                audio_tensor = torch.from_numpy(audio).unsqueeze(0)  # Add channel dimension
                torchaudio.save(str(output_path), audio_tensor, self.sample_rate)
            
            logger.debug(f"Audio saved: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Failed to save audio: {e}")
            raise RuntimeError(f"Audio saving failed: {str(e)}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "loaded": self.model_loaded,
            "sample_rate": self.sample_rate,
            "max_duration": self.max_duration,
            "memory_usage": self._get_memory_usage(),
            "capabilities": {
                "text_to_music": True,
                "audio_continuation": False,  # Not fully implemented
                "melody_conditioning": False,  # Not implemented
                "multi_track": False
            }
        }
    
    def _get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage"""
        memory_info = {}
        
        if self.device == "cuda" and torch.cuda.is_available():
            memory_info["gpu_allocated_gb"] = torch.cuda.memory_allocated() / 1e9
            memory_info["gpu_reserved_gb"] = torch.cuda.memory_reserved() / 1e9
        
        import psutil
        process = psutil.Process()
        memory_info["ram_used_gb"] = process.memory_info().rss / 1e9
        
        return memory_info
    
    def clear_cache(self):
        """Clear model cache and free memory"""
        try:
            if self.device == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Force garbage collection
            gc.collect()
            
            logger.info("Model cache cleared")
            
        except Exception as e:
            logger.warning(f"Cache clear failed: {e}")
    
    def unload_model(self):
        """Unload model to free memory"""
        try:
            if self.model is not None:
                del self.model
                self.model = None
            
            if self.processor is not None:
                del self.processor
                self.processor = None
            
            self.model_loaded = False
            
            # Clear cache
            self.clear_cache()
            
            logger.info("Model unloaded successfully")
            
        except Exception as e:
            logger.error(f"Model unload failed: {e}")
    
    def cleanup(self):
        """Cleanup resources"""
        self.unload_model()
    
    def __del__(self):
        """Destructor"""
        try:
            self.cleanup()
        except:
            pass

# Utility functions

def validate_prompt(prompt: str) -> str:
    """Validate and clean music generation prompt"""
    if not prompt or not prompt.strip():
        raise ValueError("Prompt cannot be empty")
    
    # Clean up the prompt
    cleaned = prompt.strip()
    
    # Limit length
    max_length = 200
    if len(cleaned) > max_length:
        cleaned = cleaned[:max_length].rsplit(' ', 1)[0] + "..."
        logger.warning(f"Prompt truncated to {max_length} characters")
    
    return cleaned

def estimate_generation_time(duration: float, device: str = "cpu") -> float:
    """Estimate generation time based on duration and device"""
    # Rough estimates based on empirical testing
    if device == "cuda":
        # ~1:1 ratio on modern GPU
        return duration * 1.2
    elif device == "mps":
        # ~2:1 ratio on Apple Silicon
        return duration * 2.5
    else:
        # ~8:1 ratio on CPU
        return duration * 8.0

def get_available_models() -> List[Dict[str, Any]]:
    """Get list of available MusicGen models"""
    return [
        {
            "name": "facebook/musicgen-small",
            "description": "Small model (300M parameters) - fastest generation",
            "parameters": "300M",
            "memory_gb": 2,
            "quality": "Good",
            "speed": "Fast",
            "recommended": True
        },
        {
            "name": "facebook/musicgen-medium",
            "description": "Medium model (1.5B parameters) - balanced quality/speed",
            "parameters": "1.5B",
            "memory_gb": 8,
            "quality": "Better",
            "speed": "Medium",
            "recommended": False
        },
        {
            "name": "facebook/musicgen-large",
            "description": "Large model (3.3B parameters) - highest quality",
            "parameters": "3.3B",
            "memory_gb": 16,
            "quality": "Best",
            "speed": "Slow",
            "recommended": False
        }
    ]