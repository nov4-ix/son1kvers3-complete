"""
Advanced Audio Analysis Engine for Son1k v3.0
Analyzes tempo, key, energy, vocal presence, and musical characteristics
"""

import librosa
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import logging
from scipy import signal
from scipy.stats import mode
import warnings

# Suppress librosa warnings
warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)

class AudioAnalyzer:
    """Advanced audio analysis engine"""
    
    def __init__(self, sample_rate: int = 32000):
        self.sample_rate = sample_rate
        self.hop_length = 512
        self.frame_length = 2048
        self.fft_size = 4096
        
        # Key detection profiles (Krumhansl-Schmuckler)
        self.major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        self.minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
        
        # Note names
        self.note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

    def analyze_audio_file(self, file_path: str) -> Dict[str, Any]:
        """
        Complete audio analysis pipeline
        Returns comprehensive analysis dictionary
        """
        logger.info(f"Starting audio analysis: {file_path}")
        
        try:
            # Load audio file
            audio, orig_sr = librosa.load(file_path, sr=None)
            
            # Resample if needed
            if orig_sr != self.sample_rate:
                audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=self.sample_rate)
            
            # Basic file info
            file_info = self._get_file_info(file_path, audio, orig_sr)
            
            # Core analysis
            tempo_analysis = self._analyze_tempo(audio)
            key_analysis = self._analyze_key(audio)
            energy_analysis = self._analyze_energy(audio)
            vocal_analysis = self._analyze_vocals(audio)
            spectral_analysis = self._analyze_spectral(audio)
            rhythm_analysis = self._analyze_rhythm(audio)
            
            # Compile results
            analysis = {
                "file_info": file_info,
                "tempo": tempo_analysis,
                "key_guess": key_analysis,
                "energy_structure": energy_analysis,
                "vocals": vocal_analysis,
                "spectral": spectral_analysis,
                "rhythm": rhythm_analysis,
                "summary": self._generate_summary(
                    tempo_analysis, key_analysis, energy_analysis, vocal_analysis
                )
            }
            
            logger.info(f"Analysis complete: {tempo_analysis['bpm']:.1f}bpm, {key_analysis['root']}{key_analysis['scale']}")
            return analysis
            
        except Exception as e:
            logger.error(f"Audio analysis failed: {e}")
            raise Exception(f"Audio analysis failed: {str(e)}")

    def _get_file_info(self, file_path: str, audio: np.ndarray, orig_sr: int) -> Dict[str, Any]:
        """Get basic file information"""
        file_path = Path(file_path)
        return {
            "filename": file_path.name,
            "format": file_path.suffix.lower(),
            "duration_s": len(audio) / self.sample_rate,
            "samplerate": orig_sr,
            "processed_samplerate": self.sample_rate,
            "channels": 1,  # We convert to mono
            "samples": len(audio),
            "file_size_mb": file_path.stat().st_size / (1024 * 1024) if file_path.exists() else 0
        }

    def _analyze_tempo(self, audio: np.ndarray) -> Dict[str, Any]:
        """Analyze tempo using multiple methods"""
        try:
            # Method 1: Beat tracking
            onset_envelope = librosa.onset.onset_strength(
                y=audio, sr=self.sample_rate, hop_length=self.hop_length
            )
            bpm_beats, beats = librosa.beat.beat_track(
                onset_envelope=onset_envelope,
                sr=self.sample_rate,
                hop_length=self.hop_length,
                trim=False
            )
            
            # Method 2: Onset detection as fallback
            onsets = librosa.onset.onset_detect(
                y=audio, sr=self.sample_rate, hop_length=self.hop_length, units='time'
            )
            
            if len(onsets) > 2:
                onset_intervals = np.diff(onsets)
                avg_interval = np.median(onset_intervals)
                bpm_onsets = 60.0 / avg_interval if avg_interval > 0 else bpm_beats
            else:
                bpm_onsets = bpm_beats
            
            # Method 3: Autocorrelation of onset envelope
            autocorr = np.correlate(onset_envelope, onset_envelope, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            
            # Find peaks in autocorrelation
            peaks, _ = signal.find_peaks(autocorr[1:], height=np.max(autocorr) * 0.3)
            if len(peaks) > 0:
                # Convert peak position to BPM
                peak_lag = peaks[0] + 1  # +1 because we started from index 1
                bpm_autocorr = 60.0 * self.sample_rate / (peak_lag * self.hop_length)
            else:
                bpm_autocorr = bpm_beats
            
            # Choose most reliable estimate
            bpms = [bpm_beats, bpm_onsets, bpm_autocorr]
            bpms = [bpm for bpm in bpms if 60 <= bpm <= 200]  # Filter reasonable range
            
            if bpms:
                final_bpm = np.median(bpms)
            else:
                final_bpm = bpm_beats
            
            # Confidence score based on beat consistency
            beat_times = librosa.frames_to_time(beats, sr=self.sample_rate, hop_length=self.hop_length)
            if len(beat_times) > 2:
                beat_intervals = np.diff(beat_times)
                tempo_consistency = 1.0 - (np.std(beat_intervals) / np.mean(beat_intervals))
                tempo_consistency = max(0.0, min(1.0, tempo_consistency))
            else:
                tempo_consistency = 0.5
            
            return {
                "bpm": float(final_bpm),
                "confidence": float(tempo_consistency),
                "method_results": {
                    "beat_tracking": float(bpm_beats),
                    "onset_detection": float(bpm_onsets),
                    "autocorrelation": float(bpm_autocorr)
                },
                "beat_positions": beat_times.tolist() if len(beat_times) < 100 else beat_times[:100].tolist(),
                "onset_count": len(onsets)
            }
            
        except Exception as e:
            logger.warning(f"Tempo analysis failed: {e}")
            return {
                "bpm": 120.0,
                "confidence": 0.0,
                "method_results": {},
                "beat_positions": [],
                "onset_count": 0
            }

    def _analyze_key(self, audio: np.ndarray) -> Dict[str, Any]:
        """Analyze musical key using chromagram and Krumhansl-Schmuckler algorithm"""
        try:
            # Compute chromagram
            chroma = librosa.feature.chroma_cqt(
                y=audio, sr=self.sample_rate, hop_length=self.hop_length
            )
            
            # Average across time to get overall chroma profile
            chroma_profile = np.mean(chroma, axis=1)
            
            # Normalize
            chroma_profile = chroma_profile / np.sum(chroma_profile)
            
            # Calculate correlation with major and minor profiles for each root note
            major_correlations = []
            minor_correlations = []
            
            for shift in range(12):
                # Shift the chroma profile
                shifted_chroma = np.roll(chroma_profile, shift)
                
                # Calculate correlations
                major_corr = np.corrcoef(shifted_chroma, self.major_profile)[0, 1]
                minor_corr = np.corrcoef(shifted_chroma, self.minor_profile)[0, 1]
                
                major_correlations.append(major_corr if not np.isnan(major_corr) else 0)
                minor_correlations.append(minor_corr if not np.isnan(minor_corr) else 0)
            
            # Find best matches
            best_major_idx = np.argmax(major_correlations)
            best_minor_idx = np.argmax(minor_correlations)
            best_major_corr = major_correlations[best_major_idx]
            best_minor_corr = minor_correlations[best_minor_idx]
            
            # Choose major or minor based on higher correlation
            if best_major_corr > best_minor_corr:
                key_root = self.note_names[best_major_idx]
                key_scale = "major"
                confidence = best_major_corr
            else:
                key_root = self.note_names[best_minor_idx]
                key_scale = "minor"
                confidence = best_minor_corr
            
            # Ensure confidence is in valid range
            confidence = max(0.0, min(1.0, confidence))
            
            return {
                "root": key_root,
                "scale": key_scale,
                "confidence": float(confidence),
                "chroma_profile": chroma_profile.tolist(),
                "all_correlations": {
                    "major": dict(zip(self.note_names, major_correlations)),
                    "minor": dict(zip(self.note_names, minor_correlations))
                }
            }
            
        except Exception as e:
            logger.warning(f"Key analysis failed: {e}")
            return {
                "root": "C",
                "scale": "major",
                "confidence": 0.0,
                "chroma_profile": [],
                "all_correlations": {"major": {}, "minor": {}}
            }

    def _analyze_energy(self, audio: np.ndarray) -> Dict[str, Any]:
        """Analyze energy structure and dynamics"""
        try:
            # Compute RMS energy
            rms = librosa.feature.rms(y=audio, hop_length=self.hop_length)[0]
            
            # Convert to time axis
            times = librosa.frames_to_time(
                np.arange(len(rms)), sr=self.sample_rate, hop_length=self.hop_length
            )
            
            # Smooth for structure analysis
            from scipy.ndimage import gaussian_filter1d
            rms_smooth = gaussian_filter1d(rms, sigma=2.0)
            
            # Find energy sections (simple approach)
            energy_threshold = np.percentile(rms_smooth, 60)
            high_energy_mask = rms_smooth > energy_threshold
            
            # Identify sections
            sections = []
            in_high_energy = False
            section_start = 0
            
            for i, is_high in enumerate(high_energy_mask):
                if is_high and not in_high_energy:
                    in_high_energy = True
                    section_start = times[i]
                elif not is_high and in_high_energy:
                    in_high_energy = False
                    sections.append({
                        "start": float(section_start),
                        "end": float(times[i]),
                        "type": "high_energy"
                    })
            
            # Overall statistics
            energy_stats = {
                "mean": float(np.mean(rms)),
                "std": float(np.std(rms)),
                "max": float(np.max(rms)),
                "min": float(np.min(rms)),
                "dynamic_range": float(np.max(rms) - np.min(rms))
            }
            
            return {
                "energy_curve": rms.tolist(),
                "energy_times": times.tolist(),
                "statistics": energy_stats,
                "sections": sections,
                "high_energy_percentage": float(np.mean(high_energy_mask))
            }
            
        except Exception as e:
            logger.warning(f"Energy analysis failed: {e}")
            return {
                "energy_curve": [],
                "energy_times": [],
                "statistics": {},
                "sections": [],
                "high_energy_percentage": 0.5
            }

    def _analyze_vocals(self, audio: np.ndarray) -> Dict[str, Any]:
        """Detect vocal presence using multiple features"""
        try:
            # Compute spectral features
            spectral_centroid = librosa.feature.spectral_centroid(
                y=audio, sr=self.sample_rate, hop_length=self.hop_length
            )[0]
            
            mfccs = librosa.feature.mfcc(
                y=audio, sr=self.sample_rate, n_mfcc=13, hop_length=self.hop_length
            )
            
            zero_crossing_rate = librosa.feature.zero_crossing_rate(
                y=audio, hop_length=self.hop_length
            )[0]
            
            # Vocal presence indicators
            # 1. Spectral centroid in vocal range (typically 500-4000 Hz)
            centroid_mean = np.mean(spectral_centroid)
            vocal_freq_indicator = 0.5 < centroid_mean / self.sample_rate < 0.25
            
            # 2. MFCC variance (vocals tend to have more variation)
            mfcc_variance = np.var(mfccs[1:5], axis=1)  # Use MFCC 2-5
            mfcc_indicator = np.mean(mfcc_variance) > 0.1
            
            # 3. Zero crossing rate (moderate for vocals)
            zcr_mean = np.mean(zero_crossing_rate)
            zcr_indicator = 0.01 < zcr_mean < 0.3
            
            # 4. Harmonic vs percussive content
            harmonic, percussive = librosa.effects.hpss(audio)
            harmonic_strength = np.mean(np.abs(harmonic))
            percussive_strength = np.mean(np.abs(percussive))
            
            if harmonic_strength + percussive_strength > 0:
                harmonic_ratio = harmonic_strength / (harmonic_strength + percussive_strength)
                harmonic_indicator = harmonic_ratio > 0.6  # Vocals are more harmonic
            else:
                harmonic_ratio = 0.5
                harmonic_indicator = False
            
            # Combine indicators with weights
            indicators = [vocal_freq_indicator, mfcc_indicator, zcr_indicator, harmonic_indicator]
            weights = [0.3, 0.3, 0.2, 0.2]
            
            vocal_probability = sum(w * i for w, i in zip(weights, indicators))
            has_vocals = vocal_probability > 0.5
            
            return {
                "has_vocals": bool(has_vocals),
                "vocal_probability": float(vocal_probability),
                "features": {
                    "spectral_centroid_hz": float(centroid_mean),
                    "mfcc_variance": float(np.mean(mfcc_variance)),
                    "zero_crossing_rate": float(zcr_mean),
                    "harmonic_ratio": float(harmonic_ratio)
                },
                "indicators": {
                    "frequency_range": bool(vocal_freq_indicator),
                    "mfcc_variation": bool(mfcc_indicator),
                    "zcr_range": bool(zcr_indicator),
                    "harmonic_content": bool(harmonic_indicator)
                }
            }
            
        except Exception as e:
            logger.warning(f"Vocal analysis failed: {e}")
            return {
                "has_vocals": False,
                "vocal_probability": 0.0,
                "features": {},
                "indicators": {}
            }

    def _analyze_spectral(self, audio: np.ndarray) -> Dict[str, Any]:
        """Analyze spectral characteristics"""
        try:
            # Spectral features
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(
                y=audio, sr=self.sample_rate, hop_length=self.hop_length
            ))
            
            spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(
                y=audio, sr=self.sample_rate, hop_length=self.hop_length
            ))
            
            spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(
                y=audio, sr=self.sample_rate, hop_length=self.hop_length
            ))
            
            return {
                "centroid_hz": float(spectral_centroid),
                "bandwidth_hz": float(spectral_bandwidth),
                "rolloff_hz": float(spectral_rolloff),
                "brightness": float(spectral_centroid / (self.sample_rate / 2))  # Normalized
            }
            
        except Exception as e:
            logger.warning(f"Spectral analysis failed: {e}")
            return {
                "centroid_hz": 0.0,
                "bandwidth_hz": 0.0,
                "rolloff_hz": 0.0,
                "brightness": 0.0
            }

    def _analyze_rhythm(self, audio: np.ndarray) -> Dict[str, Any]:
        """Analyze rhythmic characteristics"""
        try:
            # Onset detection
            onsets = librosa.onset.onset_detect(
                y=audio, sr=self.sample_rate, hop_length=self.hop_length, units='time'
            )
            
            # Rhythm regularity
            if len(onsets) > 2:
                onset_intervals = np.diff(onsets)
                rhythm_regularity = 1.0 - (np.std(onset_intervals) / np.mean(onset_intervals))
                rhythm_regularity = max(0.0, min(1.0, rhythm_regularity))
            else:
                rhythm_regularity = 0.0
            
            return {
                "onset_count": len(onsets),
                "rhythm_regularity": float(rhythm_regularity),
                "onset_density": float(len(onsets) / (len(audio) / self.sample_rate))
            }
            
        except Exception as e:
            logger.warning(f"Rhythm analysis failed: {e}")
            return {
                "onset_count": 0,
                "rhythm_regularity": 0.0,
                "onset_density": 0.0
            }

    def _generate_summary(self, tempo_analysis: Dict, key_analysis: Dict, 
                         energy_analysis: Dict, vocal_analysis: Dict) -> Dict[str, Any]:
        """Generate a high-level summary of the analysis"""
        
        # Categorize tempo
        bpm = tempo_analysis.get("bpm", 120)
        if bpm < 80:
            tempo_category = "slow"
        elif bpm < 120:
            tempo_category = "moderate"
        elif bpm < 160:
            tempo_category = "fast"
        else:
            tempo_category = "very_fast"
        
        # Energy level
        energy_stats = energy_analysis.get("statistics", {})
        energy_level = "medium"
        if energy_stats:
            dynamic_range = energy_stats.get("dynamic_range", 0.1)
            if dynamic_range > 0.2:
                energy_level = "high"
            elif dynamic_range < 0.05:
                energy_level = "low"
        
        # Musical characteristics
        characteristics = []
        
        if vocal_analysis.get("has_vocals", False):
            characteristics.append("vocal")
        
        if energy_level == "high":
            characteristics.append("energetic")
        elif energy_level == "low":
            characteristics.append("calm")
        
        if tempo_category in ["fast", "very_fast"]:
            characteristics.append("upbeat")
        elif tempo_category == "slow":
            characteristics.append("mellow")
        
        return {
            "tempo_category": tempo_category,
            "energy_level": energy_level,
            "key_signature": f"{key_analysis.get('root', 'C')} {key_analysis.get('scale', 'major')}",
            "has_vocals": vocal_analysis.get("has_vocals", False),
            "characteristics": characteristics,
            "overall_confidence": float(np.mean([
                tempo_analysis.get("confidence", 0),
                key_analysis.get("confidence", 0),
                vocal_analysis.get("vocal_probability", 0)
            ]))
        }

# Convenience function for direct use
def analyze_demo(file_path: str, sample_rate: int = 32000) -> Dict[str, Any]:
    """
    Convenience function to analyze a demo file
    """
    analyzer = AudioAnalyzer(sample_rate=sample_rate)
    return analyzer.analyze_audio_file(file_path)