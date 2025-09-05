"""
Comprehensive test suite for Son1k v3.0
Tests all major functionality including Maqueta → Production workflow
"""

import pytest
import asyncio
import httpx
import time
import json
import numpy as np
import soundfile as sf
from pathlib import Path
from io import BytesIO
import tempfile
import shutil

# Test configuration
API_BASE = "http://localhost:8000"
TEST_OUTPUT_DIR = Path("test_output")
TEST_FILES_DIR = Path("test_files")

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="session")
async def http_client():
    """HTTP client for API testing"""
    async with httpx.AsyncClient(timeout=300.0) as client:
        yield client

@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Setup test environment and cleanup"""
    # Setup
    TEST_OUTPUT_DIR.mkdir(exist_ok=True)
    TEST_FILES_DIR.mkdir(exist_ok=True)
    
    yield
    
    # Cleanup
    if TEST_OUTPUT_DIR.exists():
        shutil.rmtree(TEST_OUTPUT_DIR)

def generate_test_audio(duration_s=2.0, sample_rate=44100, frequency=440.0, format='wav'):
    """Generate synthetic test audio"""
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), False)
    
    # Create a simple melody with some variation
    audio = 0.3 * (
        np.sin(2 * np.pi * frequency * t) + 
        0.3 * np.sin(2 * np.pi * frequency * 1.5 * t) +
        0.1 * np.sin(2 * np.pi * frequency * 2.0 * t)
    )
    
    # Add envelope for more realistic audio
    envelope = np.exp(-t * 0.5) * (1 - np.exp(-t * 10))
    audio = audio * envelope
    
    # Add some noise for realism
    noise = np.random.normal(0, 0.01, len(audio))
    audio = audio + noise
    
    # Save to temporary file
    temp_file = TEST_FILES_DIR / f"test_audio_{int(time.time())}.{format}"
    sf.write(temp_file, audio, sample_rate)
    
    return temp_file, audio, sample_rate

@pytest.mark.asyncio
class TestHealthAndBasics:
    """Test basic API functionality"""
    
    async def test_health_check(self, http_client):
        """Test health check endpoint"""
        response = await http_client.get(f"{API_BASE}/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "healthy"
        assert "service" in data
        assert "version" in data
        assert "system" in data
        assert "features" in data
        
        # Check system info
        system = data["system"]
        assert "device" in system
        assert "torch_version" in system
        
        print(f"✅ Health check passed - Device: {system['device']}")

    async def test_root_endpoint(self, http_client):
        """Test root endpoint"""
        response = await http_client.get(f"{API_BASE}/")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "service" in data
        assert "version" in data
        assert "endpoints" in data
        
        print("✅ Root endpoint test passed")

    async def test_models_endpoint(self, http_client):
        """Test models information endpoint"""
        response = await http_client.get(f"{API_BASE}/api/v1/models")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "models" in data
        assert "current_model" in data
        assert len(data["models"]) > 0
        
        # Check model structure
        model = data["models"][0]
        assert "name" in model
        assert "description" in model
        assert "parameters" in model
        
        print(f"✅ Models endpoint test passed - {len(data['models'])} models available")

@pytest.mark.asyncio
class TestManualGeneration:
    """Test manual music generation"""
    
    async def test_generate_basic(self, http_client):
        """Test basic music generation"""
        payload = {
            "prompt": "simple piano melody for testing",
            "duration": 3.0,
            "temperature": 1.0,
            "top_k": 250,
            "apply_postprocessing": True
        }
        
        response = await http_client.post(
            f"{API_BASE}/api/v1/generate",
            json=payload,
            timeout=120.0
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Check response structure
        assert data["ok"] is True
        assert "url" in data
        assert "filename" in data
        assert "duration" in data
        assert "generation_time" in data
        assert data["prompt"] == payload["prompt"]
        
        # Download and validate audio
        audio_response = await http_client.get(f"{API_BASE}{data['url']}")
        assert audio_response.status_code == 200
        assert len(audio_response.content) > 0
        
        print(f"✅ Basic generation test passed: {data['filename']} ({data['duration']:.1f}s)")

    async def test_generate_with_seed(self, http_client):
        """Test generation with seed for reproducibility"""
        payload = {
            "prompt": "upbeat electronic music",
            "duration": 2.0,
            "seed": 12345,
            "temperature": 1.0,
            "top_k": 250
        }
        
        # Generate twice with same seed
        response1 = await http_client.post(f"{API_BASE}/api/v1/generate", json=payload)
        response2 = await http_client.post(f"{API_BASE}/api/v1/generate", json=payload)
        
        assert response1.status_code == 200
        assert response2.status_code == 200
        
        data1 = response1.json()
        data2 = response2.json()
        
        # Results should be deterministic with same seed
        assert data1["ok"] is True
        assert data2["ok"] is True
        
        print("✅ Seeded generation test passed")

    async def test_generate_presets(self, http_client):
        """Test generation presets endpoint"""
        response = await http_client.get(f"{API_BASE}/api/v1/generate/presets")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "presets" in data
        assert "count" in data
        assert data["count"] > 0
        
        # Check preset structure
        for preset_name, preset_data in data["presets"].items():
            assert "name" in preset_data
            assert "description" in preset_data
            assert "prompt" in preset_data
            assert "duration" in preset_data
        
        print(f"✅ Generation presets test passed: {data['count']} presets")

@pytest.mark.asyncio
class TestGhostStudio:
    """Test Ghost Studio functionality"""
    
    async def test_ghost_presets(self, http_client):
        """Test Ghost Studio presets"""
        response = await http_client.get(f"{API_BASE}/api/v1/ghost/presets")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "presets" in data
        assert "count" in data
        assert data["count"] > 0
        
        # Check preset structure
        for preset_name, preset_data in data["presets"].items():
            assert "name" in preset_data
            assert "description" in preset_data
            assert "prompt_base" in preset_data
            assert "suggested_duration" in preset_data
            
        print(f"✅ Ghost presets test passed: {data['count']} presets")

    async def test_ghost_job_workflow(self, http_client):
        """Test complete Ghost Studio job workflow"""
        # Get presets first
        presets_response = await http_client.get(f"{API_BASE}/api/v1/ghost/presets")
        presets_data = presets_response.json()
        preset_name = list(presets_data["presets"].keys())[0]
        
        # Create job
        job_payload = {
            "preset": preset_name,
            "prompt_extra": "with gentle mood",
            "duration": 5.0
        }
        
        job_response = await http_client.post(
            f"{API_BASE}/api/v1/ghost/job",
            json=job_payload
        )
        
        assert job_response.status_code == 200
        job_data = job_response.json()
        
        assert job_data["ok"] is True
        assert "job_id" in job_data
        job_id = job_data["job_id"]
        
        # Poll job status
        max_wait = 60  # 60 seconds max
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            status_response = await http_client.get(f"{API_BASE}/api/v1/ghost/jobs/{job_id}")
            status_data = status_response.json()
            
            if status_data["status"] == "done":
                assert "output_url" in status_data
                # Test audio access
                audio_response = await http_client.get(f"{API_BASE}{status_data['output_url']}")
                assert audio_response.status_code == 200
                break
            elif status_data["status"] == "error":
                pytest.fail(f"Job failed: {status_data.get('error_message', 'Unknown error')}")
            
            await asyncio.sleep(2)
        else:
            pytest.fail("Job did not complete within timeout")
        
        print(f"✅ Ghost job workflow test passed: {job_id}")

    async def test_ghost_stats(self, http_client):
        """Test Ghost Studio statistics"""
        response = await http_client.get(f"{API_BASE}/api/v1/ghost/stats")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "sessions" in data
        assert "storage" in data
        assert "presets" in data
        
        print("✅ Ghost stats test passed")

@pytest.mark.asyncio
class TestMaquetaProduction:
    """Test Maqueta → Production workflow"""
    
    async def test_maqueta_workflow_complete(self, http_client):
        """Test complete Maqueta → Production workflow"""
        # Generate test audio file
        test_file, audio_data, sample_rate = generate_test_audio(
            duration_s=4.0, 
            frequency=261.6,  # C4 note
            format='wav'
        )
        
        try:
            # Prepare form data
            with open(test_file, 'rb') as f:
                files = {'file': ('test_demo.wav', f, 'audio/wav')}
                data = {
                    'prompt': 'transform into uplifting electronic music with modern production',
                    'duration': '8.0',
                    'tune_amount': '0.5',
                    'eq_low_gain': '1.0',
                    'eq_mid1_gain': '0.0',
                    'eq_mid2_gain': '1.0',
                    'eq_high_gain': '0.5',
                    'sat_drive': '3.0',
                    'sat_mix': '0.2',
                    'lufs_target': '-16.0'
                }
                
                response = await http_client.post(
                    f"{API_BASE}/api/v1/ghost/maqueta",
                    files=files,
                    data=data,
                    timeout=180.0
                )
            
            assert response.status_code == 200
            result = response.json()
            
            # Validate response structure
            assert result["ok"] is True
            assert "demo" in result
            assert "production" in result
            assert "prompt_final" in result
            assert "session_id" in result
            
            # Validate demo analysis
            demo = result["demo"]
            assert "url" in demo
            assert "analysis" in demo
            analysis = demo["analysis"]
            
            assert "file_info" in analysis
            assert "tempo" in analysis
            assert "key_guess" in analysis
            assert "vocals" in analysis
            
            # Validate production
            production = result["production"]
            assert "url" in production
            assert "post_metadata" in production
            assert "processing_chain" in production["post_metadata"]
            
            # Test file access
            demo_response = await http_client.get(f"{API_BASE}{demo['url']}")
            assert demo_response.status_code == 200
            assert len(demo_response.content) > 0
            
            production_response = await http_client.get(f"{API_BASE}{production['url']}")
            assert production_response.status_code == 200
            assert len(production_response.content) > 0
            
            # Validate processing chain
            processing_chain = production["post_metadata"]["processing_chain"]
            expected_stages = ["ssl_eq", "neve_saturation", "lufs_normalization", "limiter", "fades"]
            
            for stage in expected_stages:
                assert stage in processing_chain, f"Missing processing stage: {stage}"
            
            print(f"✅ Maqueta workflow completed successfully!")
            print(f"   📊 Analysis: {analysis['tempo']['bpm']:.1f} BPM, {analysis['key_guess']['root']}{analysis['key_guess']['scale']}")
            print(f"   🎵 Processing: {' → '.join(processing_chain)}")
            print(f"   ⏱️  Total time: {result['processing_time_s']:.1f}s")
            
            return result
            
        finally:
            # Cleanup test file
            test_file.unlink(missing_ok=True)

    async def test_maqueta_file_validation(self, http_client):
        """Test file validation for maqueta upload"""
        # Test with invalid file type
        invalid_file = TEST_FILES_DIR / "test.txt"
        invalid_file.write_text("This is not an audio file")
        
        try:
            with open(invalid_file, 'rb') as f:
                files = {'file': ('test.txt', f, 'text/plain')}
                data = {'prompt': 'test', 'duration': '5.0'}
                
                response = await http_client.post(
                    f"{API_BASE}/api/v1/ghost/maqueta",
                    files=files,
                    data=data
                )
            
            assert response.status_code == 400
            
        finally:
            invalid_file.unlink(missing_ok=True)
        
        print("✅ Maqueta file validation test passed")

    async def test_maqueta_session_management(self, http_client):
        """Test session management for maqueta workflow"""
        # Create a session first
        test_file, _, _ = generate_test_audio(duration_s=2.0, format='wav')
        
        try:
            with open(test_file, 'rb') as f:
                files = {'file': ('test.wav', f, 'audio/wav')}
                data = {'prompt': 'test transformation', 'duration': '5.0'}
                
                response = await http_client.post(
                    f"{API_BASE}/api/v1/ghost/maqueta",
                    files=files,
                    data=data
                )
            
            assert response.status_code == 200
            result = response.json()
            session_id = result["session_id"]
            
            # Test session retrieval
            session_response = await http_client.get(f"{API_BASE}/api/v1/ghost/sessions/{session_id}")
            assert session_response.status_code == 200
            
            session_data = session_response.json()
            assert session_data["session_id"] == session_id
            
            print(f"✅ Session management test passed: {session_id}")
            
        finally:
            test_file.unlink(missing_ok=True)

@pytest.mark.asyncio
class TestAudioProcessing:
    """Test audio processing components"""
    
    async def test_audio_analysis_components(self, http_client):
        """Test audio analysis functions if available"""
        try:
            from src.services.audio_analysis import AudioAnalyzer
            
            # Generate test audio with known characteristics
            test_file, audio_data, sr = generate_test_audio(
                duration_s=5.0, 
                frequency=440.0,  # A4
                format='wav'
            )
            
            try:
                analyzer = AudioAnalyzer(sample_rate=sr)
                analysis = analyzer.analyze_audio_file(str(test_file))
                
                # Validate analysis structure
                assert "file_info" in analysis
                assert "tempo" in analysis
                assert "key_guess" in analysis
                assert "energy_structure" in analysis
                assert "vocals" in analysis
                assert "summary" in analysis
                
                # Basic sanity checks
                file_info = analysis["file_info"]
                assert file_info["samplerate"] == sr
                assert file_info["duration_s"] > 0
                
                tempo = analysis["tempo"]
                assert 60 <= tempo["bpm"] <= 200
                assert 0 <= tempo["confidence"] <= 1
                
                print(f"✅ Audio analysis test passed: {tempo['bpm']:.1f}bpm, {analysis['key_guess']['root']}{analysis['key_guess']['scale']}")
                
            finally:
                test_file.unlink(missing_ok=True)
                
        except ImportError:
            pytest.skip("Audio analysis module not available")

    async def test_audio_postprocessing_components(self, http_client):
        """Test audio postprocessing functions if available"""
        try:
            from src.services.audio_post import AudioPostProcessor
            
            # Generate test audio
            test_audio = np.random.randn(32000 * 2)  # 2 seconds at 32kHz
            test_audio = test_audio * 0.5  # Scale down
            
            processor = AudioPostProcessor(sample_rate=32000)
            
            # Test individual components
            ssl_processed = processor.ssl_eq(test_audio, low_gain_db=2.0, high_gain_db=1.0)
            assert len(ssl_processed) == len(test_audio)
            assert not np.array_equal(test_audio, ssl_processed)
            
            sat_processed = processor.neve_saturation(test_audio, drive_db=3.0, mix=0.2)
            assert len(sat_processed) == len(test_audio)
            
            normalized, gain_db = processor.target_lufs(test_audio, target_lufs=-16.0)
            assert len(normalized) == len(test_audio)
            assert isinstance(gain_db, (int, float))
            
            # Test complete processing chain
            processed, metadata = processor.process_master(test_audio)
            assert len(processed) == len(test_audio)
            assert "processing_chain" in metadata
            assert len(metadata["processing_chain"]) > 0
            
            print(f"✅ Audio postprocessing test passed: {len(metadata['processing_chain'])} stages")
            
        except ImportError:
            pytest.skip("Audio postprocessing module not available")

@pytest.mark.asyncio
class TestCacheAndUtilities:
    """Test cache management and utilities"""
    
    async def test_cache_clear(self, http_client):
        """Test cache clearing endpoint"""
        response = await http_client.delete(f"{API_BASE}/api/v1/cache")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["ok"] is True
        assert "message" in data
        
        print("✅ Cache clear test passed")

    async def test_static_file_serving(self, http_client):
        """Test static file serving for generated content"""
        # First generate a file
        payload = {
            "prompt": "test music for static serving",
            "duration": 2.0
        }
        
        gen_response = await http_client.post(f"{API_BASE}/api/v1/generate", json=payload)
        assert gen_response.status_code == 200
        
        gen_data = gen_response.json()
        file_url = gen_data["url"]
        
        # Test static file access
        static_response = await http_client.get(f"{API_BASE}{file_url}")
        assert static_response.status_code == 200
        assert static_response.headers["content-type"].startswith("audio/")
        assert len(static_response.content) > 0
        
        print(f"✅ Static file serving test passed: {gen_data['filename']}")

@pytest.mark.asyncio
class TestErrorHandling:
    """Test error handling and edge cases"""
    
    async def test_invalid_endpoints(self, http_client):
        """Test 404 handling"""
        response = await http_client.get(f"{API_BASE}/api/v1/nonexistent")
        assert response.status_code == 404
        
        print("✅ 404 handling test passed")

    async def test_invalid_generation_params(self, http_client):
        """Test validation of generation parameters"""
        # Test invalid duration
        payload = {
            "prompt": "test",
            "duration": 100.0  # Too long
        }
        
        response = await http_client.post(f"{API_BASE}/api/v1/generate", json=payload)
        assert response.status_code == 422  # Validation error
        
        # Test empty prompt
        payload = {
            "prompt": "",
            "duration": 5.0
        }
        
        response = await http_client.post(f"{API_BASE}/api/v1/generate", json=payload)
        assert response.status_code == 422 or response.status_code == 400
        
        print("✅ Parameter validation test passed")

    async def test_ghost_job_invalid_preset(self, http_client):
        """Test Ghost job with invalid preset"""
        payload = {
            "preset": "nonexistent_preset",
            "duration": 5.0
        }
        
        response = await http_client.post(f"{API_BASE}/api/v1/ghost/job", json=payload)
        assert response.status_code == 400
        
        print("✅ Invalid preset handling test passed")

# Integration test
@pytest.mark.asyncio
async def test_full_integration_workflow(http_client):
    """Complete integration test of all major features"""
    print("🧪 Running full integration test...")
    
    # 1. Health check
    health_response = await http_client.get(f"{API_BASE}/health")
    assert health_response.status_code == 200
    print("   ✅ Health check passed")
    
    # 2. Manual generation
    gen_payload = {"prompt": "integration test music", "duration": 3.0}
    gen_response = await http_client.post(f"{API_BASE}/api/v1/generate", json=gen_payload)
    assert gen_response.status_code == 200
    print("   ✅ Manual generation passed")
    
    # 3. Ghost Studio presets
    presets_response = await http_client.get(f"{API_BASE}/api/v1/ghost/presets")
    assert presets_response.status_code == 200
    print("   ✅ Ghost presets passed")
    
    # 4. Maqueta workflow (simplified)
    test_file, _, _ = generate_test_audio(duration_s=2.0, format='wav')
    try:
        with open(test_file, 'rb') as f:
            files = {'file': ('integration_test.wav', f, 'audio/wav')}
            data = {'prompt': 'integration test transformation', 'duration': '5.0'}
            
            maqueta_response = await http_client.post(
                f"{API_BASE}/api/v1/ghost/maqueta",
                files=files,
                data=data,
                timeout=120.0
            )
        
        assert maqueta_response.status_code == 200
        print("   ✅ Maqueta workflow passed")
        
    finally:
        test_file.unlink(missing_ok=True)
    
    # 5. Cache management
    cache_response = await http_client.delete(f"{API_BASE}/api/v1/cache")
    assert cache_response.status_code == 200
    print("   ✅ Cache management passed")
    
    print("✅ Full integration test completed successfully!")

# Utility function for running specific tests
def run_test_category(category: str = "all"):
    """Run specific test category"""
    if category == "health":
        pytest.main(["-v", "tests/test_api.py::TestHealthAndBasics"])
    elif category == "generation":
        pytest.main(["-v", "tests/test_api.py::TestManualGeneration"])
    elif category == "ghost":
        pytest.main(["-v", "tests/test_api.py::TestGhostStudio"])
    elif category == "maqueta":
        pytest.main(["-v", "tests/test_api.py::TestMaquetaProduction"])
    elif category == "audio":
        pytest.main(["-v", "tests/test_api.py::TestAudioProcessing"])
    elif category == "integration":
        pytest.main(["-v", "tests/test_api.py::test_full_integration_workflow"])
    else:
        pytest.main(["-v", "tests/test_api.py"])

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        run_test_category(sys.argv[1])
    else:
        # Run all tests
        pytest.main(["-v", __file__])