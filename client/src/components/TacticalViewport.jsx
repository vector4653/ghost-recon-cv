import React, { useState, useRef, useEffect, useCallback } from 'react';
import axios from 'axios';
import { Upload, ImageIcon, Loader2, GripVertical, AlertTriangle } from 'lucide-react';
import { API_BASE } from '../config';

const models = [
  { key: 'Baseline', label: 'Baseline (Off-the-shelf)' },
  { key: 'Exp M', label: 'Exp M (Military)' },
  { key: 'Exp C1', label: 'Exp C1 (CAMO)' },
  { key: 'Exp C2', label: 'Exp C2 (COD10K)' },
  { key: 'Exp MC1', label: 'Exp MC1 (Military + CAMO) — best' },
  { key: 'Exp MC2', label: 'Exp MC2 (Military + COD10K)' },
  { key: 'Exp MCC', label: 'Exp MCC (Tri-Domain Fusion)' },
  { key: 'MC1 GhostNet', label: 'GhostNet-SINet (MC1) — 4.6× smaller' },
  { key: 'MC1 ShuffleNet', label: 'ShuffleNetV2-SINet (MC1) — 17× smaller' },
];

export default function TacticalViewport() {
  const [selectedModel, setSelectedModel] = useState('Exp MC1');
  const [vizMode, setVizMode] = useState('both');
  const [blendOpacity, setBlendOpacity] = useState(85);
  const [sliderPos, setSliderPos] = useState(50);
  const [isDragging, setIsDragging] = useState(false);

  const [rawImageUrl, setRawImageUrl] = useState(null);
  const [maskImageUrl, setMaskImageUrl] = useState(null);
  const [overlayLayers, setOverlayLayers] = useState(null);
  const [telemetry, setTelemetry] = useState(null);
  const [latencyMs, setLatencyMs] = useState(null);
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
    setLatencyMs(null);
    setError(null);
    setSliderPos(50);
    setLoading(true);
    try {
      const formData = new FormData();
      formData.append('image', blob, filename);
      formData.append('model', modelKey || selectedModel);
      const res = await axios.post(`${API_BASE}/api/infer`, formData);
      const layers = {
        both: `data:image/png;base64,${res.data.overlayBase64 || res.data.maskBase64}`,
        outline: `data:image/png;base64,${res.data.outlineBase64 || res.data.maskBase64}`,
        fill: `data:image/png;base64,${res.data.fillBase64 || res.data.maskBase64}`,
      };
      setOverlayLayers(layers);
      setMaskImageUrl(layers[vizMode] || layers.both);
      setTelemetry(res.data.telemetry);
      setLatencyMs(typeof res.data.executionTimeMs === 'number' ? res.data.executionTimeMs : null);
    } catch (err) {
      setError('Inference backend unreachable — ensure the Express server (:5050) and FastAPI sidecar (:8000) are running.');
    } finally {
      setLoading(false);
    }
  }, [selectedModel, vizMode]);

  useEffect(() => {
    if (overlayLayers) {
      setMaskImageUrl(overlayLayers[vizMode] || overlayLayers.both);
    }
  }, [vizMode, overlayLayers]);

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


  const threatColors = {
    CRITICAL: 'var(--color-tactical-crimson)',
    ELEVATED: 'var(--color-tactical-amber)',
    LOW: 'var(--color-tactical-emerald)',
    NONE: 'var(--text-muted)',
  };
  const threatColor = threatColors[telemetry?.threatLevel] || 'var(--text-muted)';
  const engineIsSynthetic = telemetry?.inferenceEngine === 'synthetic';

  // Every row below is computed from the prediction for the image currently on
  // screen — no dataset-level constants. Benchmark scores live on the Results page.
  const num = (value, digits, suffix = '') =>
    typeof value === 'number' && Number.isFinite(value) ? `${value.toFixed(digits)}${suffix}` : '—';

  const metrics = [
    { label: 'Confidence', value: num(telemetry?.confidenceScore, 1, '%') },
    { label: 'Peak confidence', value: num(telemetry?.peakConfidence, 1, '%') },
    { label: 'Camouflage area', value: num(telemetry?.camouflageAreaPercent, 2, '%') },
    { label: 'Regions detected', value: num(telemetry?.detectedRegions, 0) },
    { label: 'Largest region', value: num(telemetry?.largestRegionPercent, 2, '%') },
    { label: 'Edge definition', value: num(telemetry?.edgeDefinition, 3) },
    { label: 'Ambiguity', value: num(telemetry?.ambiguityPercent, 2, '%') },
    { label: 'Latency', value: num(latencyMs ?? telemetry?.inferenceMs, 0, ' ms') },
  ];

  return (
    <div className="flex flex-col lg:flex-row gap-6">
      {/* Sidebar */}
      <aside className="w-full lg:w-72 shrink-0 flex flex-col gap-4">
        <section className="panel p-4 flex flex-col gap-4">
          <div>
            <label htmlFor="model" className="eyebrow block mb-1.5">Model</label>
            <select id="model" value={selectedModel} onChange={(e) => setSelectedModel(e.target.value)} className="field">
              {models.map(m => <option key={m.key} value={m.key}>{m.label}</option>)}
            </select>
          </div>

          <div>
            <label className="eyebrow block mb-1.5">Visualization Layer</label>
            <div className="grid grid-cols-3 gap-1">
              {[
                { id: 'both', label: 'Outline + Fill' },
                { id: 'outline', label: 'Outline Only' },
                { id: 'fill', label: 'Fill Only' },
              ].map(mode => (
                <button
                  key={mode.id}
                  type="button"
                  onClick={() => setVizMode(mode.id)}
                  className={`text-xs py-1.5 px-1 rounded transition-colors text-center ${vizMode === mode.id ? 'font-medium' : ''}`}
                  style={{
                    background: vizMode === mode.id ? 'var(--color-tactical-emerald)' : 'rgba(255,255,255,0.06)',
                    color: vizMode === mode.id ? '#0b0f14' : 'var(--text-muted)'
                  }}
                >
                  {mode.label}
                </button>
              ))}
            </div>
          </div>

          <div>
            <label htmlFor="opacity" className="flex justify-between mb-1.5">
              <span className="eyebrow">Mask opacity</span>
              <span className="num text-xs" style={{ color: 'var(--text-muted)' }}>{blendOpacity}%</span>
            </label>
            <input
              id="opacity"
              type="range" min="0" max="100"
              value={blendOpacity}
              onChange={(e) => setBlendOpacity(Number(e.target.value))}
              className="w-full"
              style={{ accentColor: 'var(--color-tactical-emerald)' }}
            />
          </div>

          <input ref={fileInputRef} type="file" accept="image/*" hidden onChange={handleFileChange} />
          <button onClick={() => fileInputRef.current?.click()} className="btn w-full">
            <Upload className="w-4 h-4" /> Upload image
          </button>

          <div>
            <p className="eyebrow mb-1.5">Sample images</p>
            {presets.length > 0 ? (
              <div className="flex flex-wrap gap-1.5">
                {presets.map((p) => (
                  <button
                    key={p.filename}
                    onClick={() => handlePresetClick(p)}
                    className="chip"
                    aria-pressed={activeLabel === p.label}
                  >
                    {p.label}
                  </button>
                ))}
              </div>
            ) : (
              <p className="text-xs" style={{ color: 'var(--text-faint)' }}>
                {datasetMounted === false
                  ? 'No samples found. Add images to ./Dataset/TestDataset/Images on the server.'
                  : 'Loading samples…'}
              </p>
            )}
          </div>
        </section>

        <section className="panel p-4">
          <h2 className="panel-title mb-3">Results</h2>
          <dl className="flex flex-col gap-2.5">
            {metrics.map(({ label, value }) => (
              <div key={label} className="flex justify-between items-baseline">
                <dt className="text-xs" style={{ color: 'var(--text-muted)' }}>{label}</dt>
                <dd className="num text-sm">{value}</dd>
              </div>
            ))}
          </dl>
          {telemetry && (
            <div className="mt-4 pt-3" style={{ borderTop: '1px solid var(--border)' }}>
              <div className="flex justify-between items-center">
                <span className="text-xs" style={{ color: 'var(--text-muted)' }}>Threat level</span>
                <span className="text-xs font-semibold" style={{ color: threatColor }}>{telemetry.threatLevel}</span>
              </div>
              <p className="text-xs mt-2" style={{ color: engineIsSynthetic ? 'var(--color-tactical-amber)' : 'var(--text-faint)' }}>
                {engineIsSynthetic
                  ? 'Heuristic fallback — no checkpoint loaded'
                  : `SINet-V2 · ${telemetry.model || selectedModel}`}
              </p>
            </div>
          )}
        </section>
      </aside>

      {/* Viewport */}
      <section className="panel flex flex-col flex-1 min-w-0 overflow-hidden">
        <div className="px-4 h-11 flex justify-between items-center gap-3" style={{ borderBottom: '1px solid var(--border)' }}>
          <span className="text-sm truncate" style={{ color: 'var(--text-muted)' }}>
            {activeLabel || 'No image loaded'}
          </span>
          {hasMask && (
            <button onClick={() => setSliderPos(50)} className="text-xs shrink-0" style={{ color: 'var(--text-faint)' }}>
              Reset slider
            </button>
          )}
        </div>

        {error && (
          <div className="mx-4 mt-4 rounded-md p-3 flex items-start gap-2" style={{ background: 'rgba(239,68,68,0.08)', border: '1px solid rgba(239,68,68,0.3)' }}>
            <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5" style={{ color: 'var(--color-tactical-crimson)' }} />
            <p className="text-xs leading-relaxed" style={{ color: '#fca5a5' }}>{error}</p>
          </div>
        )}

        <div className="flex-grow flex items-center justify-center p-4">
          <div
            ref={viewportRef}
            className="relative w-full flex items-center justify-center rounded-md overflow-hidden select-none"
            style={{ height: 460, background: 'var(--bg)' }}
          >
            {hasImage ? (
              <>
                {/* Raw image (base layer, always fully visible) */}
                <img
                  src={rawImageUrl}
                  alt="Input"
                  className="absolute inset-0 w-full h-full object-contain pointer-events-none"
                />

                {/* Mask overlay, clipped by curtain position */}
                {hasMask && (
                  <div
                    className="absolute inset-0 pointer-events-none"
                    style={{ clipPath: `inset(0 0 0 ${sliderPos}%)` }}
                  >
                    <img
                      src={maskImageUrl}
                      alt="Detection Overlay"
                      className="absolute inset-0 w-full h-full object-contain pointer-events-none"
                      style={{ opacity: blendOpacity / 100 }}
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
                    <div className="absolute top-0 bottom-0" style={{ width: 2, background: '#fff', opacity: 0.8 }} />
                    <div
                      className="relative rounded-full flex items-center justify-center"
                      style={{ width: 28, height: 28, background: '#fff', boxShadow: '0 1px 4px rgba(0,0,0,0.4)' }}
                    >
                      <GripVertical className="w-4 h-4" style={{ color: '#0b0f14' }} />
                    </div>
                  </div>
                )}

                {hasMask && (
                  <>
                    <span className="absolute top-2 left-2 px-2 py-0.5 rounded text-xs" style={{ background: 'rgba(0,0,0,0.6)', color: '#e5e7eb', zIndex: 20 }}>
                      Original
                    </span>
                    <span className="absolute top-2 right-2 px-2 py-0.5 rounded text-xs" style={{ background: 'rgba(0,0,0,0.6)', color: 'var(--color-tactical-emerald)', zIndex: 20 }}>
                      {vizMode === 'outline' ? 'Outline Only' : vizMode === 'fill' ? 'Mask Only' : 'Detected Soldier'}
                    </span>
                  </>
                )}

                {loading && (
                  <div className="absolute inset-0 flex items-center justify-center" style={{ background: 'rgba(11,15,20,0.6)', zIndex: 40 }}>
                    <div className="flex items-center gap-2 text-sm" style={{ color: 'var(--text-muted)' }}>
                      <Loader2 className="w-4 h-4 animate-spin" /> Running inference…
                    </div>
                  </div>
                )}
              </>
            ) : (
              <div className="text-center" style={{ color: 'var(--text-faint)' }}>
                <ImageIcon className="w-10 h-10 mx-auto mb-3" strokeWidth={1.5} />
                <p className="text-sm">Upload an image or pick a sample to run detection</p>
              </div>
            )}
          </div>
        </div>
      </section>
    </div>
  );
}
