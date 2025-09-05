import React, { useState, useEffect, useCallback } from "react";

const API = import.meta.env.VITE_API_URL || "http://localhost:8000";

// Custom hook for localStorage with error handling
const useLocalStorage = (key, initialValue) => {
  const [value, setValue] = useState(() => {
    try {
      const item = window.localStorage.getItem(key);
      return item ? JSON.parse(item) : initialValue;
    } catch (error) {
      console.error(`Error reading localStorage key "${key}":`, error);
      return initialValue;
    }
  });

  const setStoredValue = useCallback((value) => {
    try {
      setValue(value);
      window.localStorage.setItem(key, JSON.stringify(value));
    } catch (error) {
      console.error(`Error saving to localStorage key "${key}":`, error);
    }
  }, [key]);

  return [value, setStoredValue];
};

// Toast notification component
const Toast = ({ message, type, onClose }) => {
  useEffect(() => {
    const timer = setTimeout(onClose, 4000);
    return () => clearTimeout(timer);
  }, [onClose]);

  const bgColor = {
    success: '#10b981',
    error: '#ef4444',
    warning: '#f59e0b',
    info: '#3b82f6'
  }[type] || '#3b82f6';

  return (
    <div style={{
      position: 'fixed',
      top: 20,
      right: 20,
      background: bgColor,
      color: 'white',
      padding: '12px 20px',
      borderRadius: 8,
      boxShadow: '0 10px 25px rgba(0,0,0,0.2)',
      zIndex: 1000,
      maxWidth: '400px',
      animation: 'slideIn 0.3s ease-out'
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <span>{message}</span>
        <button 
          onClick={onClose}
          style={{ 
            background: 'none', 
            border: 'none', 
            color: 'white', 
            fontSize: '18px',
            cursor: 'pointer',
            marginLeft: 10
          }}
        >
          ×
        </button>
      </div>
    </div>
  );
};

// Job status badge component
const StatusBadge = ({ status }) => {
  const colors = {
    queued: '#f59e0b',
    running: '#3b82f6', 
    done: '#10b981',
    error: '#ef4444'
  };

  return (
    <span style={{
      background: colors[status] || '#6b7280',
      color: 'white',
      padding: '2px 8px',
      borderRadius: 12,
      fontSize: '12px',
      fontWeight: 600
    }}>
      {status.toUpperCase()}
    </span>
  );
};

// Error Boundary component
class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null, errorInfo: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true };
  }

  componentDidCatch(error, errorInfo) {
    this.setState({
      error: error,
      errorInfo: errorInfo
    });
  }

  render() {
    if (this.state.hasError) {
      return (
        <div style={{
          padding: '20px',
          margin: '20px',
          border: '1px solid #ef4444',
          borderRadius: '8px',
          backgroundColor: '#fef2f2'
        }}>
          <h2>Something went wrong.</h2>
          <details style={{ whiteSpace: 'pre-wrap' }}>
            {this.state.error && this.state.error.toString()}
            <br />
            {this.state.errorInfo.componentStack}
          </details>
        </div>
      );
    }

    return this.props.children;
  }
}

// Main App component
function App() {
  // Theme
  const [darkMode, setDarkMode] = useLocalStorage('darkMode', false);
  
  // UI state
  const [health, setHealth] = useState(null);
  const [toast, setToast] = useState(null);
  const [activeTab, setActiveTab] = useState('maqueta');
  
  // Loading states
  const [loading, setLoading] = useState({
    health: false,
    generate: false,
    jobs: false,
    presets: false,
    maqueta: false
  });

  // Ghost Studio state
  const [presets, setPresets] = useState({});
  const [selectedPreset, setSelectedPreset] = useState('');
  const [promptExtra, setPromptExtra] = useState('');
  const [jobs, setJobs] = useState([]);
  const [jobsPolling, setJobsPolling] = useState(false);

  // Maqueta → Production state
  const [maquetaFile, setMaquetaFile] = useState(null);
  const [maquetaPrompt, setMaquetaPrompt] = useState('');
  const [maquetaResult, setMaquetaResult] = useState(null);
  const [maquetaDuration, setMaquetaDuration] = useState(12);
  const [showAdvancedParams, setShowAdvancedParams] = useState(false);
  
  // File upload improvements
  const [dragOver, setDragOver] = useState(false);
  const [audioPreview, setAudioPreview] = useState(null);
  
  // Advanced processing parameters
  const [procParams, setProcParams] = useState({
    tune_amount: 0.7,
    eq_low_gain: 1.5,
    eq_mid1_gain: -1.0,
    eq_mid2_gain: 1.5,
    eq_high_gain: 1.0,
    sat_drive: 6.0,
    sat_mix: 0.35,
    lufs_target: -14.0
  });

  // Manual generation state
  const [gPrompt, setGPrompt] = useState("latin rock with guitars, 120 bpm, uplifting");
  const [gDuration, setGDuration] = useState(8);
  const [gTemperature, setGTemperature] = useState(1.0);
  const [gTopK, setGTopK] = useState(250);
  const [gTopP, setGTopP] = useState(0.0);
  const [gSeed, setGSeed] = useState('');
  const [gUrl, setGUrl] = useState("");

  // Utility functions
  const showToast = useCallback((message, type = 'info') => {
    setToast({ message, type });
  }, []);

  const setLoadingState = useCallback((key, value) => {
    setLoading(prev => ({ ...prev, [key]: value }));
  }, []);

  // File validation
  const validateFile = useCallback((file) => {
    const maxSize = 100 * 1024 * 1024; // 100MB
    const allowedTypes = ['audio/wav', 'audio/mpeg', 'audio/flac', 'audio/x-aiff', 'audio/mp4'];
    const allowedExtensions = ['.wav', '.mp3', '.flac', '.aiff', '.m4a'];
    
    if (file.size > maxSize) {
      return 'File too large. Maximum size is 100MB';
    }
    
    const extension = '.' + file.name.split('.').pop().toLowerCase();
    if (!allowedExtensions.includes(extension) && !allowedTypes.includes(file.type)) {
      return 'Unsupported format. Use WAV, MP3, FLAC, AIFF, or M4A';
    }
    
    return null;
  }, []);

  // File selection handler
  const handleFileSelect = useCallback((file) => {
    if (!file) return;
    
    const error = validateFile(file);
    if (error) {
      showToast(error, 'error');
      return;
    }
    
    setMaquetaFile(file);
    
    // Create audio preview URL
    const previewUrl = URL.createObjectURL(file);
    setAudioPreview(previewUrl);
    
    showToast(`File selected: ${file.name}`, 'success');
  }, [validateFile, showToast]);

  // Drag & drop handlers
  const handleDragOver = useCallback((e) => {
    e.preventDefault();
    setDragOver(true);
  }, []);

  const handleDragLeave = useCallback((e) => {
    e.preventDefault();
    setDragOver(false);
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setDragOver(false);
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      handleFileSelect(files[0]);
    }
  }, [handleFileSelect]);

  // MAIN FUNCTION: Process Maqueta → Production
  const processMaqueta = useCallback(async () => {
    if (!maquetaFile) {
      showToast('Please select a file first', 'error');
      return;
    }
    
    if (!maquetaPrompt.trim()) {
      showToast('Please describe your vision', 'error');
      return;
    }

    // Additional validation
    const error = validateFile(maquetaFile);
    if (error) {
      showToast(error, 'error');
      return;
    }

    setLoadingState('maqueta', true);
    setMaquetaResult(null);
    
    // Create AbortController for timeout
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 180000); // 3 minutes timeout
    
    try {
      // Prepare FormData
      const formData = new FormData();
      formData.append('file', maquetaFile);
      formData.append('prompt', maquetaPrompt.trim());
      formData.append('duration', maquetaDuration.toString());
      
      // Add advanced parameters
      Object.entries(procParams).forEach(([key, value]) => {
        formData.append(key, value.toString());
      });

      // Show progress toast
      showToast('Uploading file and starting analysis...', 'info');

      // Make API call with timeout
      const response = await fetch(`${API}/api/v1/ghost/maqueta`, {
        method: 'POST',
        body: formData,
        signal: controller.signal
      });

      clearTimeout(timeoutId);
      
      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || `Server error: ${response.status}`);
      }

      if (!data.ok) {
        throw new Error(data.message || 'Processing failed');
      }

      // Success
      setMaquetaResult(data);
      showToast('Production complete! Check the A/B comparison below.', 'success');

    } catch (error) {
      clearTimeout(timeoutId);
      console.error('Maqueta processing error:', error);
      
      // Better error messages
      let errorMessage = 'Processing failed: ';
      if (error.name === 'AbortError') {
        errorMessage += 'Request timed out after 3 minutes. Try with a shorter audio file.';
      } else if (error.name === 'TypeError' && error.message.includes('fetch')) {
        errorMessage += 'Cannot connect to server. Please check if the API is running.';
      } else if (error.message.includes('413')) {
        errorMessage += 'File too large for server.';
      } else if (error.message.includes('415')) {
        errorMessage += 'Unsupported file format.';
      } else {
        errorMessage += error.message;
      }
      
      showToast(errorMessage, 'error');
    } finally {
      setLoadingState('maqueta', false);
    }
  }, [maquetaFile, maquetaPrompt, maquetaDuration, procParams, validateFile, showToast, setLoadingState]);

  // Cleanup function for preview URL
  useEffect(() => {
    return () => {
      if (audioPreview) {
        URL.revokeObjectURL(audioPreview);
      }
    };
  }, [audioPreview]);

  // Load presets on mount
  useEffect(() => {
    loadPresets();
    loadJobs();
  }, []);

  // Auto-refresh jobs when polling is enabled
  useEffect(() => {
    let interval;
    if (jobsPolling) {
      interval = setInterval(loadJobs, 2000);
    }
    return () => clearInterval(interval);
  }, [jobsPolling]);

  // API functions
  const loadPresets = async () => {
    setLoadingState('presets', true);
    try {
      const res = await fetch(`${API}/api/v1/ghost/presets`);
      const data = await res.json();
      if (res.ok) {
        setPresets(data.presets);
        if (!selectedPreset && Object.keys(data.presets).length > 0) {
          setSelectedPreset(Object.keys(data.presets)[0]);
        }
      } else {
        throw new Error('Failed to load presets');
      }
    } catch (error) {
      showToast(`Failed to load presets: ${error.message}`, 'error');
    } finally {
      setLoadingState('presets', false);
    }
  };

  const loadJobs = async () => {
    setLoadingState('jobs', true);
    try {
      const res = await fetch(`${API}/api/v1/ghost/jobs?limit=20`);
      const data = await res.json();
      if (res.ok) {
        setJobs(data.jobs);
        
        const hasRunningJobs = data.jobs.some(job => job.status === 'running' || job.status === 'queued');
        setJobsPolling(hasRunningJobs);
      }
    } catch (error) {
      console.error('Failed to load jobs:', error);
    } finally {
      setLoadingState('jobs', false);
    }
  };

  const createJob = async () => {
    if (!selectedPreset) {
      showToast('Please select a preset', 'error');
      return;
    }

    try {
      const preset = presets[selectedPreset];
      const res = await fetch(`${API}/api/v1/ghost/job`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          preset: selectedPreset,
          prompt_extra: promptExtra.trim(),
          duration: preset.suggested_duration
        })
      });
      
      const data = await res.json();
      if (res.ok) {
        showToast(`Job created: ${data.job_id}`, 'success');
        setPromptExtra('');
        loadJobs();
        setJobsPolling(true);
      } else {
        throw new Error(data.detail || 'Job creation failed');
      }
    } catch (error) {
      showToast(`Error: ${error.message}`, 'error');
    }
  };

  const generateManual = async () => {
    setLoadingState('generate', true);
    setGUrl('');
    
    try {
      const payload = {
        prompt: gPrompt,
        duration: gDuration,
        temperature: gTemperature,
        top_k: gTopK,
        top_p: gTopP > 0 ? gTopP : 0,
        apply_postprocessing: true
      };
      
      if (gSeed) {
        payload.seed = parseInt(gSeed);
      }

      const res = await fetch(`${API}/api/v1/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      
      const data = await res.json();
      if (res.ok) {
        setGUrl(`${API}${data.url}`);
        showToast('Music generated successfully!', 'success');
      } else {
        throw new Error(data.detail || 'Generation failed');
      }
    } catch (error) {
      showToast(`Error: ${error.message}`, 'error');
    } finally {
      setLoadingState('generate', false);
    }
  };

  const checkHealth = async () => {
    setLoadingState('health', true);
    try {
      const res = await fetch(`${API}/health`);
      const data = await res.json();
      if (res.ok) {
        setHealth(data);
        showToast('Health check successful', 'success');
      } else {
        throw new Error('Health check failed');
      }
    } catch (error) {
      showToast(`Health check failed: ${error.message}`, 'error');
      setHealth(null);
    } finally {
      setLoadingState('health', false);
    }
  };

  // Theme colors
  const themeColors = {
    bg: darkMode ? '#111827' : '#f9fafb',
    cardBg: darkMode ? '#1f2937' : 'white',
    text: darkMode ? '#f3f4f6' : '#111827',
    textSecondary: darkMode ? '#9ca3af' : '#6b7280',
    border: darkMode ? '#374151' : '#e5e7eb',
    primary: '#3b82f6',
    success: '#10b981',
    warning: '#f59e0b',
    danger: '#ef4444'
  };

  return (
    <ErrorBoundary>
      {toast && (
        <Toast 
          message={toast.message} 
          type={toast.type} 
          onClose={() => setToast(null)} 
        />
      )}

      <div style={{ 
        fontFamily: "system-ui, -apple-system, sans-serif",
        minHeight: "100vh",
        background: themeColors.bg,
        color: themeColors.text
      }}>
        {/* Header */}
        <header style={{
          background: `linear-gradient(135deg, ${themeColors.primary}, #1e40af)`,
          color: 'white',
          padding: '20px 0',
          boxShadow: '0 4px 20px rgba(0,0,0,0.1)'
        }}>
          <div style={{ 
            maxWidth: 1200, 
            margin: '0 auto', 
            padding: '0 24px',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center'
          }}>
            <h1 style={{ 
              margin: 0,
              fontSize: '28px',
              fontWeight: 700
            }}>
              🎵 Son1k Studio v3.0
            </h1>
            
            <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
              <button
                onClick={() => setDarkMode(!darkMode)}
                style={{
                  background: 'rgba(255,255,255,0.2)',
                  border: 'none',
                  borderRadius: '50%',
                  width: 40,
                  height: 40,
                  cursor: 'pointer',
                  fontSize: '18px'
                }}
                title={darkMode ? 'Switch to light mode' : 'Switch to dark mode'}
              >
                {darkMode ? '☀️' : '🌙'}
              </button>
            </div>
          </div>
        </header>

        <div style={{ 
          maxWidth: 1200, 
          margin: '0 auto', 
          padding: 24 
        }}>
          {/* Health Check */}
          <div style={{
            background: themeColors.cardBg,
            borderRadius: 16,
            padding: 24,
            marginBottom: 24,
            border: `1px solid ${themeColors.border}`
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
              <h2 style={{ margin: 0, fontSize: '20px' }}>🔍 System Health</h2>
              <button 
                onClick={checkHealth}
                disabled={loading.health}
                style={{
                  background: loading.health ? themeColors.textSecondary : themeColors.primary,
                  color: 'white',
                  border: 'none',
                  padding: '8px 16px',
                  borderRadius: 6,
                  cursor: loading.health ? 'not-allowed' : 'pointer'
                }}
              >
                {loading.health ? "Checking..." : "Check Health"}
              </button>
            </div>
            
            <pre style={{
              background: darkMode ? '#374151' : '#f9fafb',
              padding: 12,
              borderRadius: 8,
              fontSize: '12px',
              margin: 0,
              overflow: 'auto'
            }}>
              {health ? JSON.stringify(health, null, 2) : "No health data"}
            </pre>
          </div>

          {/* Main Tabs */}
          <div style={{
            background: themeColors.cardBg,
            borderRadius: 16,
            padding: 0,
            border: `1px solid ${themeColors.border}`,
            overflow: 'hidden'
          }}>
            {/* Tab Navigation */}
            <div style={{ 
              display: 'flex',
              borderBottom: `1px solid ${themeColors.border}`
            }}>
              <button
                onClick={() => setActiveTab('maqueta')}
                style={{
                  flex: 1,
                  padding: '16px 24px',
                  border: 'none',
                  background: activeTab === 'maqueta' ? themeColors.primary : 'transparent',
                  color: activeTab === 'maqueta' ? 'white' : themeColors.text,
                  fontWeight: 600,
                  cursor: 'pointer',
                  transition: 'all 0.2s ease'
                }}
              >
                🎤 Maqueta → Production
              </button>
              <button
                onClick={() => setActiveTab('ghost')}
                style={{
                  flex: 1,
                  padding: '16px 24px',
                  border: 'none',
                  background: activeTab === 'ghost' ? themeColors.primary : 'transparent',
                  color: activeTab === 'ghost' ? 'white' : themeColors.text,
                  fontWeight: 600,
                  cursor: 'pointer',
                  transition: 'all 0.2s ease'
                }}
              >
                🤖 Ghost Studio
              </button>
              <button
                onClick={() => setActiveTab('manual')}
                style={{
                  flex: 1,
                  padding: '16px 24px',
                  border: 'none',
                  background: activeTab === 'manual' ? themeColors.primary : 'transparent',
                  color: activeTab === 'manual' ? 'white' : themeColors.text,
                  fontWeight: 600,
                  cursor: 'pointer',
                  transition: 'all 0.2s ease'
                }}
              >
                🎛️ Manual Generation
              </button>
            </div>

            <div style={{ padding: 24 }}>
              {/* Maqueta → Production Tab */}
              {activeTab === 'maqueta' && (
                <div>
                  <h3 style={{ margin: '0 0 20px 0', fontSize: '18px' }}>
                    🎤 Demo → Professional Production
                  </h3>
                  
                  <div style={{ marginBottom: 24 }}>
                    <p style={{ color: themeColors.textSecondary, marginBottom: 16, fontSize: '14px' }}>
                      Upload your demo/maqueta and describe your vision. Our AI will analyze the audio and generate a professional production with SSL EQ, pitch correction, Neve saturation, and mastering.
                    </p>
                  </div>

                  {/* Enhanced File Upload with Drag & Drop */}
                  <div style={{ marginBottom: 20 }}>
                    <label style={{ display: 'block', marginBottom: 8, fontWeight: 600 }}>
                      Upload Demo (WAV, MP3, FLAC, AIFF, M4A):
                    </label>
                    
                    <div
                      onDragOver={handleDragOver}
                      onDragLeave={handleDragLeave}
                      onDrop={handleDrop}
                      onClick={() => document.getElementById('audio-file-input').click()}
                      style={{
                        width: '100%',
                        padding: '24px',
                        border: `2px dashed ${dragOver ? themeColors.primary : themeColors.border}`,
                        borderRadius: 12,
                        background: dragOver ? (darkMode ? '#1e3a8a20' : '#dbeafe') : themeColors.cardBg,
                        color: themeColors.text,
                        cursor: 'pointer',
                        textAlign: 'center',
                        transition: 'all 0.2s ease'
                      }}
                    >
                      <input
                        id="audio-file-input"
                        type="file"
                        accept=".wav,.mp3,.flac,.aiff,.m4a,audio/*"
                        onChange={(e) => handleFileSelect(e.target.files[0])}
                        style={{ display: 'none' }}
                      />
                      
                      {!maquetaFile ? (
                        <div>
                          <div style={{ fontSize: '32px', marginBottom: 8 }}>🎵</div>
                          <div style={{ fontWeight: 600, marginBottom: 4 }}>
                            {dragOver ? 'Drop your audio file here' : 'Click or drag audio file here'}
                          </div>
                          <div style={{ fontSize: '14px', color: themeColors.textSecondary }}>
                            WAV, MP3, FLAC, AIFF, M4A (max 100MB)
                          </div>
                        </div>
                      ) : (
                        <div>
                          <div style={{ fontSize: '24px', marginBottom: 8, color: themeColors.success }}>✓</div>
                          <div style={{ fontWeight: 600, marginBottom: 4 }}>
                            {maquetaFile.name}
                          </div>
                          <div style={{ fontSize: '12px', color: themeColors.textSecondary, marginBottom: 8 }}>
                            {(maquetaFile.size / 1024 / 1024).toFixed(1)}MB
                          </div>
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              setMaquetaFile(null);
                              setAudioPreview(null);
                            }}
                            style={{
                              background: themeColors.danger,
                              color: 'white',
                              border: 'none',
                              padding: '4px 12px',
                              borderRadius: 4,
                              fontSize: '12px',
                              cursor: 'pointer'
                            }}
                          >
                            Remove
                          </button>
                        </div>
                      )}
                    </div>
                    
                    {/* Audio Preview */}
                    {audioPreview && (
                      <div style={{
                        marginTop: 12,
                        padding: 12,
                        background: darkMode ? '#374151' : '#f9fafb',
                        borderRadius: 6,
                        border: `1px solid ${themeColors.border}`
                      }}>
                        <div style={{ fontSize: '14px', fontWeight: 600, marginBottom: 8 }}>
                          🎧 Preview:
                        </div>
                        <audio 
                          src={audioPreview} 
                          controls 
                          style={{ width: '100%' }}
                          preload="metadata"
                        />
                      </div>
                    )}
                  </div>

                  {/* Production Prompt */}
                  <div style={{ marginBottom: 20 }}>
                    <label style={{ display: 'block', marginBottom: 8, fontWeight: 600 }}>
                      Describe Your Vision:
                    </label>
                    <textarea
                      value={maquetaPrompt}
                      onChange={(e) => setMaquetaPrompt(e.target.value)}
                      placeholder="e.g., 'transform this into a polished pop anthem with modern production', 'make it sound like a professional jazz recording', 'give it an electronic/synthwave vibe'..."
                      style={{
                        width: '100%',
                        height: 80,
                        padding: '12px',
                        border: `2px solid ${themeColors.border}`,
                        borderRadius: 8,
                        background: themeColors.cardBg,
                        color: themeColors.text,
                        resize: 'vertical',
                        fontSize: '14px'
                      }}
                      disabled={loading.maqueta}
                    />
                  </div>

                  {/* Basic Parameters */}
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: 16, marginBottom: 20 }}>
                    <div>
                      <label style={{ display: 'block', marginBottom: 4, fontSize: '14px', fontWeight: 600 }}>
                        Duration (seconds):
                      </label>
                      <input
                        type="number"
                        min="5"
                        max="30"
                        value={maquetaDuration}
                        onChange={(e) => setMaquetaDuration(Number(e.target.value))}
                        style={{
                          width: '100%',
                          padding: '8px 12px',
                          border: `1px solid ${themeColors.border}`,
                          borderRadius: 6,
                          background: themeColors.cardBg,
                          color: themeColors.text
                        }}
                        disabled={loading.maqueta}
                      />
                    </div>
                  </div>

                  {/* Advanced Parameters Toggle */}
                  <div style={{ marginBottom: 20 }}>
                    <button
                      onClick={() => setShowAdvancedParams(!showAdvancedParams)}
                      style={{
                        background: 'transparent',
                        border: `1px solid ${themeColors.border}`,
                        color: themeColors.text,
                        padding: '8px 16px',
                        borderRadius: 6,
                        fontSize: '12px',
                        cursor: 'pointer'
                      }}
                    >
                      {showAdvancedParams ? '🔽' : '▶️'} Advanced Processing Parameters
                    </button>
                  </div>

                  {/* Advanced Parameters */}
                  {showAdvancedParams && (
                    <div style={{
                      background: darkMode ? '#374151' : '#f9fafb',
                      padding: 16,
                      borderRadius: 8,
                      marginBottom: 20,
                      border: `1px solid ${themeColors.border}`
                    }}>
                      <h4 style={{ margin: '0 0 12px 0', fontSize: '14px', color: themeColors.primary }}>
                        🎛️ Processing Parameters
                      </h4>
                      
                      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))', gap: 12 }}>
                        {Object.entries(procParams).map(([key, value]) => (
                          <div key={key}>
                            <label style={{ display: 'block', fontSize: '12px', marginBottom: 4 }}>
                              {key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}:
                            </label>
                            <input
                              type="number"
                              step="0.1"
                              value={value}
                              onChange={(e) => setProcParams({...procParams, [key]: Number(e.target.value)})}
                              style={{ 
                                width: '100%', 
                                padding: '4px 8px', 
                                fontSize: '12px', 
                                border: `1px solid ${themeColors.border}`, 
                                borderRadius: 4,
                                background: themeColors.cardBg,
                                color: themeColors.text
                              }}
                            />
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Process Button with Enhanced Loading */}
                  <button
                    onClick={processMaqueta}
                    disabled={loading.maqueta || !maquetaFile || !maquetaPrompt.trim()}
                    style={{
                      background: (loading.maqueta || !maquetaFile || !maquetaPrompt.trim()) 
                        ? themeColors.textSecondary 
                        : 'linear-gradient(135deg, #8b5cf6, #a855f7)',
                      color: 'white',
                      border: 'none',
                      padding: '16px 32px',
                      borderRadius: 8,
                      fontSize: '16px',
                      fontWeight: 700,
                      cursor: (loading.maqueta || !maquetaFile || !maquetaPrompt.trim()) 
                        ? 'not-allowed' : 'pointer',
                      marginBottom: 32,
                      width: '100%',
                      maxWidth: '300px',
                      position: 'relative',
                      overflow: 'hidden'
                    }}
                  >
                    {loading.maqueta && (
                      <div style={{
                        position: 'absolute',
                        left: 0,
                        top: 0,
                        height: '100%',
                        background: 'rgba(255,255,255,0.2)',
                        width: '0%',
                        animation: 'progress 180s linear',
                        borderRadius: '8px'
                      }} />
                    )}
                    {loading.maqueta ? "🎵 Processing... This may take up to 3 minutes" : "🚀 Generate Production"}
                  </button>

                  {/* Results Panel A/B */}
                  {maquetaResult && (
                    <div style={{
                      background: darkMode ? '#374151' : '#f0f9ff',
                      padding: 24,
                      borderRadius: 12,
                      border: `1px solid ${themeColors.border}`,
                      marginTop: 20
                    }}>
                      <h3 style={{ margin: '0 0 20px 0', color: themeColors.primary }}>
                        🎉 Production Complete - A/B Comparison
                      </h3>

                      {/* Analysis Summary */}
                      <div style={{ marginBottom: 20, fontSize: '14px' }}>
                        <div style={{
                          background: darkMode ? '#4b5563' : 'white',
                          padding: 12,
                          borderRadius: 6,
                          marginBottom: 12
                        }}>
                          <strong>Analysis Results:</strong>
                          {maquetaResult.demo?.analysis && (
                            <div style={{ marginTop: 8 }}>
                              <span>🎵 Tempo: {maquetaResult.demo.analysis.tempo?.bpm?.toFixed(1)} BPM</span>
                              <span style={{ marginLeft: 16 }}>🎼 Key: {maquetaResult.demo.analysis.key_guess?.root}{maquetaResult.demo.analysis.key_guess?.scale}</span>
                              <span style={{ marginLeft: 16 }}>🎤 Vocals: {maquetaResult.demo.analysis.vocals?.has_vocals ? 'Detected' : 'None'}</span>
                            </div>
                          )}
                        </div>
                        <div style={{
                          background: darkMode ? '#4b5563' : 'white',
                          padding: 8,
                          borderRadius: 6,
                          fontSize: '12px',
                          fontFamily: 'monospace'
                        }}>
                          <strong>AI Prompt Used:</strong> "{maquetaResult.prompt_final}"
                        </div>
                      </div>

                      {/* A/B Players */}
                      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20 }}>
                        {/* Demo (A) */}
                        <div style={{
                          background: darkMode ? '#4b5563' : 'white',
                          padding: 16,
                          borderRadius: 8,
                          border: `2px solid ${themeColors.warning}`
                        }}>
                          <h4 style={{ margin: '0 0 12px 0', fontSize: '16px', color: themeColors.warning }}>
                            🎤 A: Original Demo
                          </h4>
                          <audio 
                            src={`${API}${maquetaResult.demo.url}`} 
                            controls 
                            style={{ width: '100%' }} 
                          />
                          <div style={{ fontSize: '12px', marginTop: 8, color: themeColors.textSecondary }}>
                            Duration: {maquetaResult.demo.duration_s?.toFixed(1)}s
                          </div>
                        </div>

                        {/* Production (B) */}
                        <div style={{
                          background: darkMode ? '#4b5563' : 'white',
                          padding: 16,
                          borderRadius: 8,
                          border: `2px solid ${themeColors.success}`
                        }}>
                          <h4 style={{ margin: '0 0 12px 0', fontSize: '16px', color: themeColors.success }}>
                            🎵 B: AI Production
                          </h4>
                          <audio 
                            src={`${API}${maquetaResult.production.url}`} 
                            controls 
                            style={{ width: '100%' }} 
                          />
                          <div style={{ fontSize: '12px', marginTop: 8, color: themeColors.textSecondary }}>
                            Duration: {maquetaResult.production.duration_s?.toFixed(1)}s • 
                            Device: {maquetaResult.production.device} • 
                            Gain: {maquetaResult.production.post_metadata?.lufs_gain_db?.toFixed(1)}dB
                          </div>
                        </div>
                      </div>

                      {/* Processing Details */}
                      <div style={{ 
                        marginTop: 16, 
                        padding: 12, 
                        background: darkMode ? '#6b7280' : '#f3f4f6',
                        borderRadius: 6,
                        fontSize: '12px'
                      }}>
                        <strong>Processing Chain Applied:</strong> {maquetaResult.production.post_metadata?.processing_chain?.join(' → ') || 'Unknown'}
                        <div style={{ marginTop: 4 }}>
                          <strong>Session ID:</strong> {maquetaResult.session_id} • 
                          <strong>Processing Time:</strong> {maquetaResult.processing_time_s?.toFixed(1)}s
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              )}

              {/* Ghost Studio Tab */}
              {activeTab === 'ghost' && (
                <div>
                  <h3 style={{ margin: '0 0 20px 0', fontSize: '18px' }}>
                    🤖 Automated Music Generation
                  </h3>
                  
                  {/* Preset Selection */}
                  <div style={{ marginBottom: 20 }}>
                    <label style={{ display: 'block', marginBottom: 8, fontWeight: 600 }}>
                      Select Preset:
                    </label>
                    <select
                      value={selectedPreset}
                      onChange={(e) => setSelectedPreset(e.target.value)}
                      style={{
                        width: '100%',
                        padding: '12px',
                        border: `1px solid ${themeColors.border}`,
                        borderRadius: 8,
                        background: themeColors.cardBg,
                        color: themeColors.text,
                        fontSize: '14px'
                      }}
                    >
                      {Object.entries(presets).map(([key, preset]) => (
                        <option key={key} value={key}>
                          {preset.name} - {preset.description}
                        </option>
                      ))}
                    </select>
                  </div>

                  {/* Current Preset Info */}
                  {selectedPreset && presets[selectedPreset] && (
                    <div style={{
                      background: darkMode ? '#374151' : '#f0f9ff',
                      padding: 16,
                      borderRadius: 8,
                      marginBottom: 20,
                      border: `1px solid ${themeColors.border}`
                    }}>
                      <h4 style={{ margin: '0 0 8px 0', color: themeColors.primary }}>
                        {presets[selectedPreset].name}
                      </h4>
                      <p style={{ margin: '0 0 8px 0', fontSize: '14px', color: themeColors.textSecondary }}>
                        {presets[selectedPreset].description}
                      </p>
                      <p style={{ margin: 0, fontSize: '12px', fontFamily: 'monospace' }}>
                        <strong>Base prompt:</strong> {presets[selectedPreset].prompt_base}
                      </p>
                      <div style={{ marginTop: 8, fontSize: '12px', color: themeColors.textSecondary }}>
                        <span>🎵 {presets[selectedPreset].suggested_bpm} BPM</span>
                        <span style={{ marginLeft: 16 }}>⏱️ {presets[selectedPreset].suggested_duration}s</span>
                        <span style={{ marginLeft: 16 }}>🏷️ {presets[selectedPreset].tags?.join(', ')}</span>
                      </div>
                    </div>
                  )}

                  {/* Extra Prompt */}
                  <div style={{ marginBottom: 20 }}>
                    <label style={{ display: 'block', marginBottom: 8, fontWeight: 600 }}>
                      Additional Instructions (optional):
                    </label>
                    <textarea
                      value={promptExtra}
                      onChange={(e) => setPromptExtra(e.target.value)}
                      placeholder="Add specific details to customize the generation..."
                      style={{
                        width: '100%',
                        height: 60,
                        padding: '12px',
                        border: `1px solid ${themeColors.border}`,
                        borderRadius: 8,
                        background: themeColors.cardBg,
                        color: themeColors.text,
                        resize: 'vertical'
                      }}
                    />
                  </div>

                  {/* Create Job Button */}
                  <button
                    onClick={createJob}
                    disabled={!selectedPreset}
                    style={{
                      background: !selectedPreset ? themeColors.textSecondary : themeColors.success,
                      color: 'white',
                      border: 'none',
                      padding: '16px 32px',
                      borderRadius: 8,
                      fontSize: '16px',
                      fontWeight: 700,
                      cursor: !selectedPreset ? 'not-allowed' : 'pointer',
                      marginBottom: 32
                    }}
                  >
                    🚀 Create Ghost Job
                  </button>

                  {/* Jobs List */}
                  <div>
                    <div style={{ 
                      display: 'flex', 
                      justifyContent: 'space-between', 
                      alignItems: 'center',
                      marginBottom: 16 
                    }}>
                      <h4 style={{ margin: 0, fontSize: '16px' }}>📋 Recent Jobs</h4>
                      <button
                        onClick={loadJobs}
                        disabled={loading.jobs}
                        style={{
                          background: 'transparent',
                          border: `1px solid ${themeColors.border}`,
                          color: themeColors.text,
                          padding: '6px 12px',
                          borderRadius: 4,
                          fontSize: '12px',
                          cursor: 'pointer'
                        }}
                      >
                        {loading.jobs ? "Refreshing..." : "🔄 Refresh"}
                      </button>
                    </div>

                    <div style={{ maxHeight: '400px', overflowY: 'auto' }}>
                      {jobs.length === 0 ? (
                        <p style={{ color: themeColors.textSecondary, textAlign: 'center', padding: 20 }}>
                          No jobs yet. Create your first Ghost job!
                        </p>
                      ) : (
                        jobs.map((job) => (
                          <div key={job.id} style={{
                            background: darkMode ? '#374151' : '#f9fafb',
                            padding: 16,
                            borderRadius: 8,
                            marginBottom: 8,
                            border: `1px solid ${themeColors.border}`
                          }}>
                            <div style={{ 
                              display: 'flex', 
                              justifyContent: 'space-between', 
                              alignItems: 'flex-start',
                              marginBottom: 8 
                            }}>
                              <div>
                                <div style={{ fontWeight: 600, marginBottom: 4 }}>
                                  {presets[job.preset]?.name || job.preset}
                                </div>
                                <div style={{ fontSize: '12px', color: themeColors.textSecondary }}>
                                  {new Date(job.created_at).toLocaleString()}
                                </div>
                              </div>
                              <StatusBadge status={job.status} />
                            </div>
                            
                            {job.prompt_extra && (
                              <div style={{ fontSize: '12px', margin: '8px 0', fontStyle: 'italic' }}>
                                "+ {job.prompt_extra}"
                              </div>
                            )}
                            
                            {job.output_url && (
                              <audio 
                                src={`${API}${job.output_url}`} 
                                controls 
                                style={{ width: '100%', marginTop: 8 }} 
                              />
                            )}
                            
                            {job.error_message && (
                              <div style={{ 
                                color: themeColors.danger, 
                                fontSize: '12px', 
                                marginTop: 8,
                                padding: 8,
                                background: darkMode ? '#7f1d1d' : '#fef2f2',
                                borderRadius: 4
                              }}>
                                Error: {job.error_message}
                              </div>
                            )}
                          </div>
                        ))
                      )}
                    </div>
                  </div>
                </div>
              )}

              {/* Manual Generation Tab */}
              {activeTab === 'manual' && (
                <div>
                  <h3 style={{ margin: '0 0 20px 0', fontSize: '18px' }}>
                    🎛️ Manual Music Generation
                  </h3>
                  
                  {/* Main Prompt */}
                  <div style={{ marginBottom: 20 }}>
                    <label style={{ display: 'block', marginBottom: 8, fontWeight: 600 }}>
                      Music Description:
                    </label>
                    <textarea
                      value={gPrompt}
                      onChange={(e) => setGPrompt(e.target.value)}
                      style={{
                        width: '100%',
                        height: 80,
                        padding: '12px',
                        border: `1px solid ${themeColors.border}`,
                        borderRadius: 8,
                        background: themeColors.cardBg,
                        color: themeColors.text,
                        resize: 'vertical'
                      }}
                    />
                  </div>

                  {/* Parameters Grid */}
                  <div style={{ 
                    display: 'grid', 
                    gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', 
                    gap: 16,
                    marginBottom: 24 
                  }}>
                    <div>
                      <label style={{ display: 'block', marginBottom: 4, fontSize: '14px', fontWeight: 600 }}>
                        Duration (seconds):
                      </label>
                      <input
                        type="number"
                        min="1"
                        max="30"
                        value={gDuration}
                        onChange={(e) => setGDuration(Number(e.target.value))}
                        style={{
                          width: '100%',
                          padding: '8px 12px',
                          border: `1px solid ${themeColors.border}`,
                          borderRadius: 6,
                          background: themeColors.cardBg,
                          color: themeColors.text
                        }}
                      />
                    </div>

                    <div>
                      <label style={{ display: 'block', marginBottom: 4, fontSize: '14px', fontWeight: 600 }}>
                        Temperature:
                      </label>
                      <input
                        type="number"
                        min="0.1"
                        max="2.0"
                        step="0.1"
                        value={gTemperature}
                        onChange={(e) => setGTemperature(Number(e.target.value))}
                        style={{
                          width: '100%',
                          padding: '8px 12px',
                          border: `1px solid ${themeColors.border}`,
                          borderRadius: 6,
                          background: themeColors.cardBg,
                          color: themeColors.text
                        }}
                      />
                    </div>

                    <div>
                      <label style={{ display: 'block', marginBottom: 4, fontSize: '14px', fontWeight: 600 }}>
                        Top-K:
                      </label>
                      <input
                        type="number"
                        min="1"
                        max="1000"
                        value={gTopK}
                        onChange={(e) => setGTopK(Number(e.target.value))}
                        style={{
                          width: '100%',
                          padding: '8px 12px',
                          border: `1px solid ${themeColors.border}`,
                          borderRadius: 6,
                          background: themeColors.cardBg,
                          color: themeColors.text
                        }}
                      />
                    </div>

                    <div>
                      <label style={{ display: 'block', marginBottom: 4, fontSize: '14px', fontWeight: 600 }}>
                        Top-P:
                      </label>
                      <input
                        type="number"
                        min="0"
                        max="1"
                        step="0.05"
                        value={gTopP}
                        onChange={(e) => setGTopP(Number(e.target.value))}
                        style={{
                          width: '100%',
                          padding: '8px 12px',
                          border: `1px solid ${themeColors.border}`,
                          borderRadius: 6,
                          background: themeColors.cardBg,
                          color: themeColors.text
                        }}
                      />
                    </div>

                    <div>
                      <label style={{ display: 'block', marginBottom: 4, fontSize: '14px', fontWeight: 600 }}>
                        Seed (optional):
                      </label>
                      <input
                        type="text"
                        value={gSeed}
                        onChange={(e) => setGSeed(e.target.value)}
                        placeholder="Random if empty"
                        style={{
                          width: '100%',
                          padding: '8px 12px',
                          border: `1px solid ${themeColors.border}`,
                          borderRadius: 6,
                          background: themeColors.cardBg,
                          color: themeColors.text
                        }}
                      />
                    </div>
                  </div>

                  {/* Generate Button */}
                  <button
                    onClick={generateManual}
                    disabled={loading.generate || !gPrompt.trim()}
                    style={{
                      background: loading.generate ? themeColors.textSecondary : themeColors.warning,
                      color: 'white',
                      border: 'none',
                      padding: '16px 32px',
                      borderRadius: 8,
                      fontSize: '16px',
                      fontWeight: 700,
                      cursor: loading.generate ? 'not-allowed' : 'pointer',
                      marginBottom: 20
                    }}
                  >
                    {loading.generate ? "🎵 Generating..." : "🚀 Generate Music"}
                  </button>

                  {/* Generated Audio */}
                  {gUrl && (
                    <div style={{
                      background: darkMode ? '#374151' : '#f0f9ff',
                      padding: 20,
                      borderRadius: 12,
                      border: `1px solid ${themeColors.border}`
                    }}>
                      <h4 style={{ margin: '0 0 12px 0' }}>🎶 Generated Music</h4>
                      <audio 
                        src={gUrl} 
                        controls 
                        style={{ width: '100%' }} 
                      />
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* CSS for animations */}
      <style jsx>{`
        @keyframes slideIn {
          from { 
            opacity: 0;
            transform: translateX(100%);
          }
          to { 
            opacity: 1;
            transform: translateX(0);
          }
        }
        
        @keyframes progress {
          from { width: 0%; }
          to { width: 100%; }
        }
      `}</style>
    </ErrorBoundary>
  );
}

export default App;