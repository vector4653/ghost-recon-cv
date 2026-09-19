import React, { useState, useRef, useEffect, useCallback } from 'react';
import axios from 'axios';
import {
  Target, Upload, Activity, ShieldAlert, Sliders, Play, Camera,
  Loader2, GripVertical, AlertTriangle, MapPin
} from 'lucide-react';
import { API_BASE } from '../config';

const models = [
  { key: 'Baseline', label: 'Baseline (Off-the-shelf)' },
  { key: 'Exp M', label: 'Exp M (Military)' },
  { key: 'Exp C1', label: 'Exp C1 (CAMO)' },
  { key: 'Exp C2', label: 'Exp C2 (COD10K)' },
  { key: 'Exp MC1', label: 'Exp MC1 (Winner)' },
  { key: 'Exp MC2', label: 'Exp MC2 (Military + COD10K)' },
  { key: 'Exp MCC', label: 'Exp MCC (Tri-Domain Fusion)' },
];

export default function TacticalViewport() {
  const [selectedModel, setSelectedModel] = useState('Exp MC1');
  const [blendOpacity, setBlendOpacity] = useState(55);
  const [sliderPos, setSliderPos] = useState(50);
  const [isDragging, setIsDragging] = useState(false);

  const [rawImageUrl, setRawImageUrl] = useState(null);
  const [maskImageUrl, setMaskImageUrl] = useState(null);
  const [telemetry, setTelemetry] = useState(null);
  const [activeLabel, setActiveLabel] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const [presets, setPresets] = useState([]);
  const [datasetMounted, setDatasetMounted] = useState(true);

  const fileInputRef = useRef(null);
  const viewportRef = useRef(null);
  const currentSourceRef = useRef(null);
  const isFirstRun = useRef(true);

  useEffect(() => {
    axios.get(`${API_BASE}/api/presets`)
      .then((res) => {
        setPresets(res.data.presets || []);
        setDatasetMounted(res.data.mounted !== false);
      })
      .catch(() => {
        setPresets([]);
        setDatasetMounted(false);
      });
  }, []);

  const processSource = useCallback(async (blob, filename, label, modelKey) => {
    setRawImageUrl(URL.createObjectURL(blob));
    setActiveLabel(label || filename);
    currentSourceRef.current = { blob, filename, label: label || filename };
    setMaskImageUrl(null);
    setTelemetry(null);
    setError(null);
    setSliderPos(50);
    setLoading(true);
    try {
      const formData = new FormData();
      formData.append('image', blob, filename);
      formData.append('model', modelKey || selectedModel);
      const res = await axios.post(`${API_BASE}/api/infer`, formData);
      setMaskImageUrl(`data:image/png;base64,${res.data.maskBase64}`);
      setTelemetry(res.data.telemetry);
    } catch (err) {
      setError('Inference backend unreachable — ensure the Express server (:5000) and FastAPI sidecar (:8000) are running.');
    } finally {
      setLoading(false);
    }
  }, [selectedModel]);

  const handleFileChange = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    processSource(file, file.name, file.name);
    e.target.value = '';
  };

  const handlePresetClick = async (preset) => {
    try {
      const res = await axios.get(`${API_BASE}${preset.url}`, { responseType: 'blob' });
      processSource(res.data, preset.filename, preset.label);
    } catch (err) {
      setError('Could not load preset image from the dataset mount.');
    }
  };

  // Re-run inference against the same source when the model selection changes
  useEffect(() => {
    if (isFirstRun.current) {
      isFirstRun.current = false;
      return;
    }
    if (currentSourceRef.current) {
      const { blob, filename, label } = currentSourceRef.current;
      processSource(blob, filename, label, selectedModel);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedModel]);

  // Curtain drag handling
  const updateSliderFromClientX = useCallback((clientX) => {
    if (!viewportRef.current) return;
    const rect = viewportRef.current.getBoundingClientRect();
    const pct = ((clientX - rect.left) / rect.width) * 100;
    setSliderPos(Math.min(100, Math.max(0, pct)));
  }, []);

  useEffect(() => {
    if (!isDragging) return;
    const onMove = (e) => updateSliderFromClientX(e.clientX ?? e.touches?.[0]?.clientX);
    const onUp = () => setIsDragging(false);
    window.addEventListener('pointermove', onMove);
    window.addEventListener('pointerup', onUp);
    return () => {
      window.removeEventListener('pointermove', onMove);
      window.removeEventListener('pointerup', onUp);
    };
  }, [isDragging, updateSliderFromClientX]);

  const hasImage = Boolean(rawImageUrl);
  const hasMask = Boolean(maskImageUrl);

  const threatColor = telemetry?.threatLevel === 'CRITICAL' ? 'var(--color-tactical-crimson)' : 'var(--color-tactical-amber)';

  return (
    <div className="flex flex-col lg:flex-row gap-6 p-6" style={{ minHeight: 'calc(100vh - 100px)' }}>
      {/* Left Sidebar */}
      <div className="w-full lg:w-72 flex flex-col gap-5">

        {/* Operational Parameters */}
        <div className="glass-panel p-5 relative overflow-hidden">
          <div className="scanline"></div>
          <h2 className="flex items-center gap-2 font-bold mb-4 uppercase tracking-wider text-sm" style={{ color: 'var(--color-tactical-emerald)' }}>
            <Target className="w-4 h-4" /> Operational Parameters
          </h2>

          <div className="mb-5">
            <label className="block text-xs uppercase tracking-widest mb-2" style={{ color: '#94a3b8' }}>Model Selection</label>
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              className="w-full text-sm rounded p-2 outline-none"
              style={{
                backgroundColor: '#0a0d12',
                border: '1px solid #334155',
                color: '#e2e8f0',
              }}
            >
              {models.map(m => <option key={m.key} value={m.key}>{m.label}</option>)}
            </select>
          </div>

          <div className="mb-5">
            <label className="flex justify-between text-xs uppercase tracking-widest mb-2">
              <span style={{ color: '#94a3b8' }}>Mask Opacity</span>
              <span style={{ color: 'var(--color-tactical-cyan)' }}>{blendOpacity}%</span>
            </label>
            <input
              type="range" min="0" max="100"
              value={blendOpacity}
              onChange={(e) => setBlendOpacity(Number(e.target.value))}
              className="w-full"
              style={{ accentColor: 'var(--color-tactical-cyan)' }}
            />
          </div>

          <input ref={fileInputRef} type="file" accept="image/*" hidden onChange={handleFileChange} />
          <button
            onClick={() => fileInputRef.current?.click()}
            className="w-full text-sm uppercase tracking-wider py-3 rounded flex justify-center items-center gap-2 transition-colors"
            style={{ background: '#1e293b', border: '1px solid #334155', color: '#94a3b8' }}
            onMouseEnter={e => e.currentTarget.style.background = '#273449'}
            onMouseLeave={e => e.currentTarget.style.background = '#1e293b'}
          >
            <Upload className="w-4 h-4" /> Upload Tactical Feed
          </button>

          {/* Preset Scenarios */}
          <div className="mt-5">
            <label className="block text-xs uppercase tracking-widest mb-2" style={{ color: '#94a3b8' }}>Preset Scenarios</label>
            {presets.length > 0 ? (
              <div className="grid grid-cols-2 gap-2">
                {presets.map((p) => (
                  <button
                    key={p.filename}
                    onClick={() => handlePresetClick(p)}
                    className="text-left px-2 py-2 rounded text-xs transition-colors flex items-center gap-1.5"
                    style={{
                      background: activeLabel === p.label ? 'rgba(16,185,129,0.12)' : 'rgba(0,0,0,0.3)',
                      border: `1px solid ${activeLabel === p.label ? 'rgba(16,185,129,0.4)' : '#1e293b'}`,
                      color: activeLabel === p.label ? 'var(--color-tactical-emerald)' : '#94a3b8',
                    }}
                  >
                    <MapPin className="w-3 h-3 flex-shrink-0" />
                    <span className="truncate">{p.label}</span>
                  </button>
                ))}
              </div>
            ) : (
              <div className="rounded p-2.5 flex items-start gap-2" style={{ background: 'rgba(0,0,0,0.3)', border: '1px solid #1e293b' }}>
                <AlertTriangle className="w-3.5 h-3.5 flex-shrink-0 mt-0.5" style={{ color: '#475569' }} />
                <p className="text-xs leading-relaxed" style={{ color: '#475569' }}>
                  {datasetMounted === false
                    ? 'Dataset mount not found. Presets require ./Dataset/TestDataset/Images on the server.'
                    : 'Loading presets…'}
                </p>
              </div>
            )}
          </div>
        </div>

        {/* Live Telemetry */}
        <div className="glass-panel p-5 flex-grow">
          <h2 className="flex items-center gap-2 font-bold mb-4 uppercase tracking-wider text-sm" style={{ color: 'var(--color-tactical-amber)' }}>
            <Activity className="w-4 h-4" /> Live Telemetry
          </h2>

          <div className="space-y-3">
            {[
              { label: 'Detection Confidence', value: telemetry ? `${telemetry.confidenceScore.toFixed(1)}%` : '—', color: 'var(--color-tactical-emerald)' },
              { label: 'Camouflage Area', value: telemetry ? `${telemetry.camouflageAreaPercent.toFixed(1)}%` : '—', color: '#e2e8f0' },
              { label: 'Boundary IoU (4px)', value: telemetry?.boundaryIoU !== undefined ? telemetry.boundaryIoU.toFixed(4) : '—', color: 'var(--color-tactical-cyan)' },
              { label: 'False Alarm Rate', value: telemetry?.fpr !== undefined ? `${telemetry.fpr.toFixed(2)}%` : '—', color: 'var(--color-tactical-amber)' },
            ].map(({ label, value, color }) => (
              <div key={label} className="flex justify-between items-center pb-2" style={{ borderBottom: '1px solid #1e293b' }}>
                <span className="text-xs uppercase tracking-wide" style={{ color: '#64748b' }}>{label}</span>
                <span className="font-mono font-bold text-sm" style={{ color }}>{value}</span>
              </div>
            ))}
          </div>

          <div className="mt-4 rounded p-3 flex items-start gap-3" style={{ background: 'rgba(239,68,68,0.08)', border: '1px solid rgba(239,68,68,0.3)' }}>
            <ShieldAlert className="w-5 h-5 flex-shrink-0 mt-0.5" style={{ color: threatColor }} />
            <div>
              <p className="text-xs font-bold uppercase tracking-wider mb-1" style={{ color: threatColor }}>
                {telemetry ? `Threat Assessment — ${telemetry.threatLevel}` : 'Threat Assessment'}
              </p>
              <p className="text-xs leading-relaxed" style={{ color: '#cbd5e1' }}>
                {telemetry
                  ? `Camouflaged signature covering ${telemetry.camouflageAreaPercent.toFixed(1)}% of frame. Model confidence ${telemetry.confidenceScore.toFixed(1)}%.`
                  : 'Awaiting visual feed — upload an image or select a preset scenario to begin inference.'}
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Main Viewport */}
      <div className="glass-panel flex flex-col flex-1">
        {/* Header bar */}
        <div className="p-3 flex justify-between items-center" style={{ borderBottom: '1px solid #1e293b', background: 'rgba(0,0,0,0.4)' }}>
          <div className="flex items-center gap-3">
            <div className="w-2 h-2 rounded-full animate-pulse" style={{ background: loading ? 'var(--color-tactical-amber)' : 'var(--color-tactical-emerald)' }}></div>
            <span className="text-xs font-bold uppercase tracking-widest" style={{ color: 'var(--color-tactical-emerald)' }}>
              {loading ? 'Processing Uplink…' : 'Live Uplink Active'} — {models.find(m => m.key === selectedModel)?.label}
            </span>
          </div>
          <div className="flex gap-2">
            <button
              onClick={() => setSliderPos(50)}
              className="p-1.5 rounded transition-colors"
              style={{ color: '#64748b' }}
              title="Reset Curtain Slider"
            >
              <Sliders className="w-4 h-4" />
            </button>
            <button className="p-1.5 rounded transition-colors" style={{ color: '#64748b' }} title="Export Frame">
              <Camera className="w-4 h-4" />
            </button>
          </div>
        </div>

        {error && (
          <div className="mx-4 mt-3 rounded p-3 flex items-start gap-2" style={{ background: 'rgba(239,68,68,0.08)', border: '1px solid rgba(239,68,68,0.3)' }}>
            <AlertTriangle className="w-4 h-4 flex-shrink-0 mt-0.5" style={{ color: 'var(--color-tactical-crimson)' }} />
            <p className="text-xs leading-relaxed" style={{ color: '#fca5a5' }}>{error}</p>
          </div>
        )}

        {/* Viewport area */}
        <div className="flex-grow flex items-center justify-center p-4" style={{ background: '#05080c' }}>
          <div
            ref={viewportRef}
            className="relative w-full max-w-3xl flex items-center justify-center rounded overflow-hidden select-none"
            style={{ minHeight: 420, height: 420, border: '1px solid #1e293b', background: '#0a0d12' }}
          >
            {hasImage ? (
              <>
                {/* Raw image (base layer, always fully visible) */}
                <img
                  src={rawImageUrl}
                  alt="Tactical feed"
                  className="absolute inset-0 w-full h-full object-contain pointer-events-none"
                />

                {/* Mask overlay, clipped by curtain position */}
                {hasMask && (
                  <div
                    className="absolute inset-0 pointer-events-none"
                    style={{ clipPath: `inset(0 0 0 ${sliderPos}%)` }}
                  >
                    <div
                      className="absolute inset-0"
                      style={{
                        backgroundColor: `rgba(16, 185, 129, ${blendOpacity / 100})`,
                        WebkitMaskImage: `url(${maskImageUrl})`,
                        maskImage: `url(${maskImageUrl})`,
                        WebkitMaskSize: 'contain',
                        maskSize: 'contain',
                        WebkitMaskRepeat: 'no-repeat',
                        maskRepeat: 'no-repeat',
                        WebkitMaskPosition: 'center',
                        maskPosition: 'center',
                      }}
                    />
                  </div>
                )}

                {/* Curtain handle */}
                {hasMask && (
                  <div
                    className="absolute top-0 bottom-0 flex items-center justify-center"
                    style={{ left: `${sliderPos}%`, transform: 'translateX(-50%)', zIndex: 30, cursor: 'ew-resize' }}
                    onPointerDown={(e) => { e.preventDefault(); setIsDragging(true); }}
                  >
                    <div className="absolute top-0 bottom-0" style={{ width: 2, background: 'var(--color-tactical-cyan)', boxShadow: '0 0 8px var(--color-tactical-cyan)' }} />
                    <div
                      className="relative rounded-full flex items-center justify-center"
                      style={{ width: 32, height: 32, background: '#0a0d12', border: '2px solid var(--color-tactical-cyan)', boxShadow: '0 0 12px rgba(6,182,212,0.5)' }}
                    >
                      <GripVertical className="w-4 h-4" style={{ color: 'var(--color-tactical-cyan)' }} />
                    </div>
                  </div>
                )}

                {/* Labels */}
                <div className="absolute top-2 left-2 px-2 py-1 rounded text-xs font-bold uppercase tracking-wide" style={{ background: 'rgba(0,0,0,0.6)', color: '#94a3b8', zIndex: 20 }}>
                  Raw Feed
                </div>
                {hasMask && (
                  <div className="absolute top-2 right-2 px-2 py-1 rounded text-xs font-bold uppercase tracking-wide" style={{ background: 'rgba(16,185,129,0.15)', color: 'var(--color-tactical-emerald)', zIndex: 20 }}>
                    COD Mask Overlay
                  </div>
                )}

                {loading && (
                  <div className="absolute inset-0 flex items-center justify-center" style={{ background: 'rgba(2,4,8,0.5)', zIndex: 40 }}>
                    <div className="flex flex-col items-center gap-2">
                      <Loader2 className="w-8 h-8 animate-spin" style={{ color: 'var(--color-tactical-cyan)' }} />
                      <span className="text-xs uppercase tracking-widest" style={{ color: '#94a3b8' }}>Running Inference…</span>
                    </div>
                  </div>
                )}
              </>
            ) : (
              <>
                {/* Corner markers */}
                {['top-2 left-2', 'top-2 right-2', 'bottom-2 left-2', 'bottom-2 right-2'].map((pos, i) => (
                  <div key={i} className={`absolute ${pos} w-4 h-4`}
                    style={{
                      borderTop: i < 2 ? '2px solid rgba(6,182,212,0.5)' : 'none',
                      borderBottom: i >= 2 ? '2px solid rgba(6,182,212,0.5)' : 'none',
                      borderLeft: i % 2 === 0 ? '2px solid rgba(6,182,212,0.5)' : 'none',
                      borderRight: i % 2 === 1 ? '2px solid rgba(6,182,212,0.5)' : 'none',
                    }}
                  />
                ))}
                {/* Crosshair */}
                <div className="absolute" style={{ top: '50%', left: '50%', transform: 'translate(-50%, -50%)' }}>
                  <div className="w-8 h-8 rounded-full flex items-center justify-center" style={{ border: '1px solid rgba(6,182,212,0.3)' }}>
                    <div className="w-1.5 h-1.5 rounded-full" style={{ background: 'var(--color-tactical-cyan)' }}></div>
                  </div>
                </div>
                <div className="absolute" style={{ top: '50%', left: 16, width: 40, height: 1, background: 'rgba(6,182,212,0.25)' }} />
                <div className="absolute" style={{ top: '50%', right: 16, width: 40, height: 1, background: 'rgba(6,182,212,0.25)' }} />
                <div className="absolute" style={{ left: '50%', top: 16, width: 1, height: 40, background: 'rgba(6,182,212,0.25)' }} />
                <div className="absolute" style={{ left: '50%', bottom: 16, width: 1, height: 40, background: 'rgba(6,182,212,0.25)' }} />

                <div className="text-center" style={{ color: '#334155' }}>
                  <Play className="w-12 h-12 mx-auto mb-3" />
                  <p className="uppercase tracking-widest text-sm">Awaiting Visual Feed</p>
                  <p className="text-xs mt-1" style={{ color: '#1e293b' }}>Upload an image or select a preset scenario to begin</p>
                </div>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
