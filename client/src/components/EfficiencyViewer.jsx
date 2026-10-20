import React, { useState, useEffect } from 'react';
import axios from 'axios';
import {
  Zap,
  Cpu,
  Layers,
  Activity,
  Maximize2,
  X,
  Gauge,
  HardDrive,
  Scale,
  Sparkles,
  Info
} from 'lucide-react';
import { API_BASE } from '../config';

const fallbackEfficiency = [
  {
    "Model": "SINet-V2 Res2Net-50 (MC1)",
    "Params (M)": "26.98",
    "MACs (G)": "12.31",
    "FLOPs (G)": "24.63",
    "Size (MB)": "103.4",
    "GPU FP32 (ms)": "28.72",
    "GPU FP16 (ms)": "19.99",
    "FPS": "34.8",
    "CPU Latency (ms)": "128.9",
    "Peak VRAM (MB)": "133.95",
    "S_alpha": "0.8675",
    "E_phi": "0.9095",
    "F_beta_w": "0.7773",
    "MAE": "0.0322",
    "Boundary IoU (4px)": "0.3437",
    "FPR (%)": "1.55",
    "Param Reduction (x)": "1.0x",
    "MACs Reduction (x)": "1.0x",
    "Speedup (x)": "1.0x",
    "% F_beta_w Retained": "100.0%",
    "F_w / GMAC": "0.063",
    "S_alpha / M Params": "0.032",
    "Lightweight Criterion": "Baseline Reference"
  },
  {
    "Model": "GhostNet-SINet (MC1)",
    "Params (M)": "5.92",
    "MACs (G)": "2.26",
    "FLOPs (G)": "4.51",
    "Size (MB)": "23.1",
    "GPU FP32 (ms)": "13.55",
    "GPU FP16 (ms)": "16.90",
    "FPS": "73.8",
    "CPU Latency (ms)": "57.3",
    "Peak VRAM (MB)": "38.51",
    "S_alpha": "0.8376",
    "E_phi": "0.8590",
    "F_beta_w": "0.7078",
    "MAE": "0.0451",
    "Boundary IoU (4px)": "0.2495",
    "FPR (%)": "2.48",
    "Param Reduction (x)": "4.56x",
    "MACs Reduction (x)": "5.46x",
    "Speedup (x)": "2.12x",
    "% F_beta_w Retained": "91.06%",
    "F_w / GMAC": "0.314",
    "S_alpha / M Params": "0.142",
    "Lightweight Criterion": "NO"
  },
  {
    "Model": "ShuffleNetV2-SINet (MC1)",
    "Params (M)": "1.59",
    "MACs (G)": "1.19",
    "FLOPs (G)": "2.37",
    "Size (MB)": "6.4",
    "GPU FP32 (ms)": "10.56",
    "GPU FP16 (ms)": "13.37",
    "FPS": "94.7",
    "CPU Latency (ms)": "50.0",
    "Peak VRAM (MB)": "15.14",
    "S_alpha": "0.8294",
    "E_phi": "0.8515",
    "F_beta_w": "0.6929",
    "MAE": "0.0472",
    "Boundary IoU (4px)": "0.2503",
    "FPR (%)": "2.39",
    "Param Reduction (x)": "16.96x",
    "MACs Reduction (x)": "10.38x",
    "Speedup (x)": "2.72x",
    "% F_beta_w Retained": "89.14%",
    "F_w / GMAC": "0.584",
    "S_alpha / M Params": "0.522",
    "Lightweight Criterion": "NO"
  }
];

const figures = [
  {
    file: 'pareto_fbeta_vs_macs.png',
    title: 'Pareto Trade-off Curve',
    subtitle: 'Accuracy vs. Compute Cost',
    caption: 'Weighted F-measure (F_w) vs MACs (GigaOps). Bubble size indicates parameter footprint (M). Demonstrates detection retention per compute cost.'
  },
  {
    file: 'params_and_macs_comparison.png',
    title: 'Parameter & MAC Complexity',
    subtitle: 'Footprint Reduction',
    caption: 'Side-by-side comparison of model weights (M) and multiply-accumulate operations (G) showing extreme reductions in ShuffleNet and GhostNet.'
  },
  {
    file: 'latency_comparison.png',
    title: 'Edge Inference Latency',
    subtitle: 'NVIDIA RTX 4050 Latency & Throughput',
    caption: 'Measured latency (ms/frame) and FPS across GPU FP32 and FP16 half-precision modes with CUDA stream synchronization.'
  },
  {
    file: 'backbone_vs_decoder_breakdown.png',
    title: 'Architectural Decomposition',
    subtitle: 'Backbone vs Shared Decoder',
    caption: 'Parameter distribution showing the fixed RFB + NCD + GRA decoder footprint (~0.73M - 1.26M) vs the scalable backbone extractor.'
  },
  {
    file: 'lightweight_visual_comparison_grid.png',
    title: '6-Scenario Tactical Mask Grid',
    subtitle: 'Qualitative Silhouette Fidelity',
    caption: 'Ground truth vs predicted mask fidelity across snipers, foliage, and prone soldiers for Res2Net-50, GhostNet, and ShuffleNetV2.'
  }
];

function StatCard({ label, value, unit, sub, icon: Icon }) {
  return (
    <div className="rounded-lg p-4 flex-1 min-w-[200px]" style={{ background: 'var(--bg)', border: '1px solid var(--border)' }}>
      <div className="flex items-center justify-between mb-2">
        <p className="text-xs font-medium" style={{ color: 'var(--text-muted)' }}>{label}</p>
        {Icon && <Icon className="w-4 h-4" style={{ color: 'var(--color-tactical-emerald)' }} />}
      </div>
      <p className="num text-2xl font-bold tracking-tight text-white">
        {value}
        {unit && <span className="text-sm ml-1 font-normal" style={{ color: 'var(--text-faint)' }}>{unit}</span>}
      </p>
      {sub && <p className="text-xs mt-1" style={{ color: 'var(--text-faint)' }}>{sub}</p>}
    </div>
  );
}

export default function EfficiencyViewer() {
  const [data, setData] = useState(fallbackEfficiency);
  const [selectedFigure, setSelectedFigure] = useState(figures[0]);
  const [lightbox, setLightbox] = useState(null);

  useEffect(() => {
    axios.get(`${API_BASE}/api/efficiency`)
      .then(res => {
        if (Array.isArray(res.data) && res.data.length > 0) {
          setData(res.data);
        }
      })
      .catch(() => {
        // Use embedded fallback data
      });
  }, []);

  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <div className="flex items-center gap-2 mb-1">
          <Zap className="w-5 h-5" style={{ color: 'var(--color-tactical-emerald)' }} />
          <h1 className="text-xl font-bold tracking-tight text-white">Tactical Edge Efficiency & Lightweight Analysis</h1>
        </div>
        <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
          Benchmarking lightweight vision backbones (GhostNet & ShuffleNetV2) against Res2Net-50 under identical MC1 fine-tuning for low-power edge deployment (UAVs, soldier HUDs).
        </p>
      </div>

      {/* Highlights Bar */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          label="Max Parameter Reduction"
          value="17.0×"
          sub="ShuffleNetV2: 1.59M vs 26.98M"
          icon={Layers}
        />
        <StatCard
          label="Max Compute Reduction"
          value="10.4×"
          sub="ShuffleNetV2: 1.19 GMACs vs 12.31"
          icon={Cpu}
        />
        <StatCard
          label="Peak Edge Framerate"
          value="94.7"
          unit="FPS"
          sub="10.56 ms/frame on RTX 4050"
          icon={Gauge}
        />
        <StatCard
          label="Compact Checkpoint Size"
          value="6.36"
          unit="MB"
          sub="93.8% storage compression"
          icon={HardDrive}
        />
      </div>

      {/* Table Section */}
      <div className="rounded-lg overflow-hidden" style={{ background: 'var(--bg)', border: '1px solid var(--border)' }}>
        <div className="px-5 py-4 border-b flex items-center justify-between" style={{ borderColor: 'var(--border)' }}>
          <div>
            <h2 className="text-sm font-semibold text-white">Comparative Efficiency & Detection Matrix</h2>
            <p className="text-xs" style={{ color: 'var(--text-faint)' }}>Evaluated on the standardized 580-image Military Test split (352×352 input resolution)</p>
          </div>
          <span className="text-xs px-2.5 py-1 rounded font-mono" style={{ background: '#1a222d', color: 'var(--color-tactical-emerald)' }}>
            NVIDIA RTX 4050 Laptop GPU
          </span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr style={{ background: '#0e1217', borderBottom: '1px solid var(--border)' }}>
                <th className="py-3 px-4 text-left font-medium text-slate-300">Model</th>
                <th className="py-3 px-3 text-center font-medium text-slate-300">Params ↓</th>
                <th className="py-3 px-3 text-center font-medium text-slate-300">MACs ↓</th>
                <th className="py-3 px-3 text-center font-medium text-slate-300">Size ↓</th>
                <th className="py-3 px-3 text-center font-medium text-slate-300">GPU FP32 ↓</th>
                <th className="py-3 px-3 text-center font-medium text-slate-300">FPS ↑</th>
                <th className="py-3 px-3 text-center font-medium text-slate-300">CPU Latency ↓</th>
                <th className="py-3 px-3 text-center font-medium text-slate-300">S<sub>α</sub> ↑</th>
                <th className="py-3 px-3 text-center font-medium text-slate-300">F<sub>β</sub><sup>w</sup> ↑</th>
                <th className="py-3 px-3 text-center font-medium text-slate-300">B-IoU ↑</th>
                <th className="py-3 px-3 text-center font-medium text-slate-300">FPR ↓</th>
                <th className="py-3 px-3 text-center font-medium text-slate-300">Param Red.</th>
                <th className="py-3 px-3 text-center font-medium text-slate-300">MAC Red.</th>
                <th className="py-3 px-3 text-center font-medium text-slate-300">% F<sub>β</sub><sup>w</sup></th>
                <th className="py-3 px-4 text-center font-medium text-slate-300">Lightweight?</th>
              </tr>
            </thead>
            <tbody className="divide-y" style={{ borderColor: 'var(--border)' }}>
              {data.map((row, idx) => {
                const isBase = idx === 0;
                return (
                  <tr key={row.Model} className="hover:bg-slate-800/20 transition-colors">
                    <td className="py-3 px-4 font-semibold text-white whitespace-nowrap">
                      {row.Model}
                      {isBase && <span className="ml-2 text-[10px] px-1.5 py-0.5 rounded bg-blue-900/40 text-blue-300 border border-blue-700/50">Baseline</span>}
                    </td>
                    <td className="py-3 px-3 text-center font-mono text-slate-200">{parseFloat(row['Params (M)']).toFixed(2)}M</td>
                    <td className="py-3 px-3 text-center font-mono text-slate-200">{parseFloat(row['MACs (G)']).toFixed(2)}G</td>
                    <td className="py-3 px-3 text-center font-mono text-slate-200">{parseFloat(row['Size (MB)']).toFixed(1)}MB</td>
                    <td className="py-3 px-3 text-center font-mono text-slate-200">{parseFloat(row['GPU FP32 (ms)']).toFixed(2)}ms</td>
                    <td className="py-3 px-3 text-center font-mono font-bold" style={{ color: 'var(--color-tactical-emerald)' }}>
                      {parseFloat(row['FPS']).toFixed(1)}
                    </td>
                    <td className="py-3 px-3 text-center font-mono text-slate-300">{parseFloat(row['CPU Latency (ms)']).toFixed(1)}ms</td>
                    <td className="py-3 px-3 text-center font-mono text-slate-200">{parseFloat(row['S_alpha']).toFixed(4)}</td>
                    <td className="py-3 px-3 text-center font-mono font-bold text-white">{parseFloat(row['F_beta_w']).toFixed(4)}</td>
                    <td className="py-3 px-3 text-center font-mono text-slate-200">{parseFloat(row['Boundary IoU (4px)']).toFixed(4)}</td>
                    <td className="py-3 px-3 text-center font-mono text-slate-300">{row['FPR (%)'] || (row['fpr_background'] ? (row['fpr_background']*100).toFixed(2) : '—')}%</td>
                    <td className="py-3 px-3 text-center font-mono font-bold text-emerald-400">{row['Param Reduction (x)']}</td>
                    <td className="py-3 px-3 text-center font-mono font-bold text-emerald-400">{row['MACs Reduction (x)']}</td>
                    <td className="py-3 px-3 text-center font-mono text-slate-300">{row['% F_beta_w Retained']}</td>
                    <td className="py-3 px-4 text-center whitespace-nowrap">
                      {isBase ? (
                        <span className="text-[11px] px-2 py-0.5 rounded font-mono bg-slate-800 text-slate-300">Reference</span>
                      ) : (
                        <span className="text-[11px] px-2 py-0.5 rounded font-mono bg-amber-950/60 text-amber-300 border border-amber-800/40">
                          {row['Lightweight Criterion'] || 'NO'}
                        </span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        <div className="p-4 bg-slate-950/40 border-t flex items-start gap-2.5 text-xs text-slate-400" style={{ borderColor: 'var(--border)' }}>
          <Info className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
          <div>
            <span className="font-semibold text-slate-200">Formal Lightweight Criterion Definition:</span> A model is certified as <strong>Lightweight</strong> if and only if it achieves <strong>≥ 4.0× parameter reduction</strong> AND <strong>≥ 4.0× MACs reduction</strong> compared to the Res2Net-50 baseline while retaining <strong>≥ 95.0% of weighted F-measure (F<sub>β</sub><sup>w</sup>)</strong>. Neither GhostNet (91.06% retained) nor ShuffleNetV2 (89.14% retained) achieved the 95% retention threshold due to fine camouflage silhouette fidelity requirements.
          </div>
        </div>
      </div>

      {/* Visual Figures Studio */}
      <div className="rounded-lg p-5 space-y-4" style={{ background: 'var(--bg)', border: '1px solid var(--border)' }}>
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div>
            <h2 className="text-sm font-semibold text-white">Efficiency Visual Analytics & Diagnostics</h2>
            <p className="text-xs" style={{ color: 'var(--text-faint)' }}>High-resolution publication figures generated at 300 DPI</p>
          </div>
          {/* Tabs */}
          <div className="flex flex-wrap gap-1 bg-slate-900/60 p-1 rounded-lg border border-slate-800">
            {figures.map(fig => (
              <button
                key={fig.file}
                onClick={() => setSelectedFigure(fig)}
                className={`px-3 py-1.5 rounded-md text-xs font-medium transition-all ${
                  selectedFigure.file === fig.file
                    ? 'bg-slate-800 text-white shadow-sm'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                {fig.title}
              </button>
            ))}
          </div>
        </div>

        {/* Selected Figure Viewer */}
        <div className="rounded-lg border overflow-hidden" style={{ background: '#0b0f14', borderColor: 'var(--border)' }}>
          <div className="relative group flex items-center justify-center p-4 min-h-[380px]">
            <img
              src={`/efficiency-figures/${selectedFigure.file}`}
              alt={selectedFigure.title}
              className="max-h-[520px] w-auto object-contain rounded"
              onError={(e) => {
                // If static route isn't serving, fallback to local path or placeholder
                e.target.src = `/artifacts/${selectedFigure.file}`;
              }}
            />
            <button
              onClick={() => setLightbox(`/efficiency-figures/${selectedFigure.file}`)}
              className="absolute top-4 right-4 p-2 rounded-md bg-slate-900/80 text-slate-300 hover:text-white border border-slate-700 opacity-0 group-hover:opacity-100 transition-opacity"
              title="Expand figure"
            >
              <Maximize2 className="w-4 h-4" />
            </button>
          </div>
          <div className="px-5 py-3 border-t bg-slate-950/60 flex flex-col sm:flex-row sm:items-center justify-between gap-2" style={{ borderColor: 'var(--border)' }}>
            <div>
              <p className="text-xs font-semibold text-white">{selectedFigure.title} — <span className="font-normal text-slate-400">{selectedFigure.subtitle}</span></p>
              <p className="text-[11px] text-slate-400 mt-0.5">{selectedFigure.caption}</p>
            </div>
          </div>
        </div>
      </div>

      {/* Engineering Takeaways Grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="rounded-lg p-5 space-y-2" style={{ background: 'var(--bg)', border: '1px solid var(--border)' }}>
          <div className="flex items-center gap-2 text-white font-semibold text-sm">
            <Zap className="w-4 h-4 text-amber-400" />
            FLOPs vs. Latency Discrepancy
          </div>
          <p className="text-xs text-slate-300 leading-relaxed">
            While ShuffleNetV2 cuts theoretical operations by <strong>10.38×</strong> vs Res2Net-50 and by <strong>1.90×</strong> vs GhostNet, its GPU runtime is <strong>10.56 ms</strong> vs <strong>13.55 ms</strong> (a 1.28× speedup over GhostNet). Depthwise convolutions and channel shuffles suffer from low arithmetic intensity (operations per byte transferred), becoming memory-bandwidth bound on modern GPUs.
          </p>
        </div>

        <div className="rounded-lg p-5 space-y-2" style={{ background: 'var(--bg)', border: '1px solid var(--border)' }}>
          <div className="flex items-center gap-2 text-white font-semibold text-sm">
            <Activity className="w-4 h-4 text-emerald-400" />
            FP16 Precision Behavior
          </div>
          <p className="text-xs text-slate-300 leading-relaxed">
            Res2Net-50 drops from 28.72 ms to 19.99 ms under FP16 (<strong>1.44× FP16 speedup</strong>) because its 25.7M parameters saturate GPU tensor pipes. Lightweight backbones show small kernel launch overhead at batch size 1 (16.9 ms for GhostNet), making FP32 the preferred format for low-latency micro-batching.
          </p>
        </div>

        <div className="rounded-lg p-5 space-y-2" style={{ background: 'var(--bg)', border: '1px solid var(--border)' }}>
          <div className="flex items-center gap-2 text-white font-semibold text-sm">
            <Scale className="w-4 h-4 text-blue-400" />
            Tactical Edge Suitability
          </div>
          <p className="text-xs text-slate-300 leading-relaxed">
            Although the models didn't meet the strict 95% retention threshold (retaining 91.06% and 89.14% F<sub>β</sub><sup>w</sup>), ShuffleNetV2 delivers a micro <strong>6.36 MB checkpoint</strong>, <strong>15.1 MB VRAM usage</strong>, and <strong>94.7 FPS</strong>, making it uniquely viable for battery-constrained micro-UAV payloads.
          </p>
        </div>
      </div>

      {/* Lightbox Modal */}
      {lightbox && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/85 backdrop-blur-sm"
          onClick={() => setLightbox(null)}
        >
          <div className="relative max-w-6xl max-h-[90vh] bg-slate-900 p-2 rounded-lg border border-slate-700 shadow-2xl" onClick={e => e.stopPropagation()}>
            <button
              onClick={() => setLightbox(null)}
              className="absolute -top-3 -right-3 p-1.5 rounded-full bg-slate-800 text-slate-300 hover:text-white border border-slate-600 shadow"
            >
              <X className="w-4 h-4" />
            </button>
            <img src={lightbox} alt="Full resolution figure" className="max-h-[82vh] w-auto object-contain rounded" />
          </div>
        </div>
      )}
    </div>
  );
}
