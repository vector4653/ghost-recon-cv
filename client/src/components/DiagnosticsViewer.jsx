import React, { useState } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell
} from 'recharts';
import { Crosshair, ShieldOff, ImageIcon, X, TrendingDown, TrendingUp } from 'lucide-react';
import { API_BASE } from '../config';

const diagnostics = [
  { name: 'Baseline', short: 'Base', boundaryIoU: 0.2332, fpr: 1.41, bgLeakage: 1.58, winner: false },
  { name: 'Exp M',    short: 'M',    boundaryIoU: 0.3372, fpr: 2.02, bgLeakage: 2.13, winner: false },
  { name: 'Exp C1',   short: 'C1',   boundaryIoU: 0.2225, fpr: 1.58, bgLeakage: 1.68, winner: false },
  { name: 'Exp C2',   short: 'C2',   boundaryIoU: 0.1414, fpr: 0.81, bgLeakage: 1.14, winner: false },
  { name: 'Exp MC1',  short: 'MC1',  boundaryIoU: 0.3437, fpr: 1.55, bgLeakage: 1.67, winner: true  },
  { name: 'Exp MC2',  short: 'MC2',  boundaryIoU: 0.3119, fpr: 2.19, bgLeakage: 2.45, winner: false },
  { name: 'Exp MCC',  short: 'MCC',  boundaryIoU: 0.3199, fpr: 1.99, bgLeakage: 2.28, winner: false },
];

const winnerEntry = diagnostics.find(d => d.winner);
const baseline = diagnostics[0];
const expM = diagnostics.find(d => d.short === 'M');

const gallery = [
  {
    file: 'boundary_sharpness_analysis.png',
    title: 'Boundary Sharpness Analysis',
    caption: '4px narrow-band contour IoU comparison across all 7 experiments, isolating edge precision from bulk segmentation accuracy.',
  },
  {
    file: 'false_alarm_background_diagnostics.png',
    title: 'False Alarm & Background Diagnostics',
    caption: 'FPR and background leakage rates on non-target tactical terrain — the cost of over-triggering on foliage and clutter.',
  },
  {
    file: 'tactical_overlay_comparison.png',
    title: 'Tactical Overlay Comparison',
    caption: 'Heatmap overlays of predicted camouflage probability across operational scenarios for every model variant.',
  },
  {
    file: 'visual_comparison_grid.png',
    title: 'Visual Comparison Grid',
    caption: 'Side-by-side raw / ground-truth / predicted-mask triplets across 6 challenging operational scenarios.',
  },
];

const tooltipStyle = { backgroundColor: '#0a0d12', borderColor: '#1e293b', color: '#f8fafc', fontSize: 12 };

function StatCard({ label, value, unit, color, icon: Icon, sub }) {
  return (
    <div className="rounded p-4 flex-1 min-w-[140px]" style={{ background: 'rgba(0,0,0,0.3)', border: '1px solid #1e293b' }}>
      <div className="flex items-center gap-2 mb-1" style={{ color: '#64748b' }}>
        {Icon && <Icon className="w-3.5 h-3.5" />}
        <span className="text-xs uppercase tracking-widest">{label}</span>
      </div>
      <p className="font-mono font-bold text-2xl" style={{ color }}>{value}<span className="text-sm ml-1" style={{ color: '#475569' }}>{unit}</span></p>
      {sub && <p className="text-xs mt-1" style={{ color: '#475569' }}>{sub}</p>}
    </div>
  );
}

export default function DiagnosticsViewer() {
  const [selected, setSelected] = useState(winnerEntry.short);
  const [lightbox, setLightbox] = useState(null);
  const [failedImages, setFailedImages] = useState({});
  const active = diagnostics.find(d => d.short === selected);

  return (
    <div className="p-6 max-w-7xl mx-auto overflow-y-auto" style={{ maxHeight: 'calc(100vh - 120px)' }}>

      {/* Header */}
      <div className="glass-panel p-5 mb-6">
        <h1 className="text-xl font-bold uppercase tracking-wider mb-1" style={{ color: 'var(--color-tactical-amber)' }}>
          Qualitative Diagnostics & Error Analysis
        </h1>
        <p className="text-sm" style={{ color: '#64748b' }}>
          Phase 3 visual evidence — boundary precision, false-alarm suppression, and cross-model failure inspection.
        </p>
      </div>

      {/* Two analysis cards */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">

        {/* Boundary Sharpness Card */}
        <div className="glass-panel p-5">
          <h2 className="flex items-center gap-2 text-sm font-bold uppercase tracking-widest mb-3" style={{ color: 'var(--color-tactical-cyan)' }}>
            <Crosshair className="w-4 h-4" /> Boundary Sharpness (4px Contour IoU)
          </h2>
          <p className="text-xs leading-relaxed mb-4" style={{ color: '#94a3b8' }}>
            We isolate a narrow 4-pixel band around each ground-truth contour and measure IoU only within that band —
            a test of silhouette edge fidelity, decoupled from bulk region overlap. Sharp boundaries matter tactically:
            a soldier's outline that bleeds into foliage is a silhouette an operator can't trust.
            <strong style={{ color: 'var(--color-tactical-emerald)' }}> Exp MC1 leads at 0.3437</strong>, edging out
            Exp M (0.3372) and dwarfing the off-the-shelf Baseline (0.2332) — a
            <strong style={{ color: 'var(--color-tactical-emerald)' }}> +47.4% </strong>
            relative gain from military-only training.
          </p>
          <ResponsiveContainer width="100%" height={180}>
            <BarChart data={diagnostics} margin={{ top: 5, right: 10, left: -25, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
              <XAxis dataKey="short" stroke="#334155" tick={{ fill: '#64748b', fontSize: 11 }} />
              <YAxis stroke="#334155" tick={{ fill: '#64748b', fontSize: 11 }} domain={[0, 0.4]} />
              <Tooltip contentStyle={tooltipStyle} formatter={(v) => v.toFixed(4)} />
              <Bar dataKey="boundaryIoU" radius={[3, 3, 0, 0]}>
                {diagnostics.map((d) => (
                  <Cell key={d.short} fill={d.winner ? '#10b981' : '#06b6d4'} fillOpacity={d.winner ? 1 : 0.55} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* False Positive Card */}
        <div className="glass-panel p-5">
          <h2 className="flex items-center gap-2 text-sm font-bold uppercase tracking-widest mb-3" style={{ color: 'var(--color-tactical-crimson)' }}>
            <ShieldOff className="w-4 h-4" /> False Positive & Distractor Analysis
          </h2>
          <p className="text-xs leading-relaxed mb-4" style={{ color: '#94a3b8' }}>
            False alarms on background foliage and terrain clutter drive cognitive overload during the OODA loop's
            "Orient" phase. Fusing natural camouflage (CAMO) with military data regularizes the network against
            over-triggering: <strong style={{ color: 'var(--color-tactical-emerald)' }}>Exp MC1 cuts FPR to 1.55%</strong>,
            a <strong style={{ color: 'var(--color-tactical-emerald)' }}>~23% reduction</strong> versus military-only
            fine-tuning (Exp M, 2.02%), while background leakage drops from 2.13% to 1.67%.
          </p>
          <ResponsiveContainer width="100%" height={180}>
            <BarChart data={diagnostics} margin={{ top: 5, right: 10, left: -25, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
              <XAxis dataKey="short" stroke="#334155" tick={{ fill: '#64748b', fontSize: 11 }} />
              <YAxis stroke="#334155" tick={{ fill: '#64748b', fontSize: 11 }} domain={[0, 2.5]} unit="%" />
              <Tooltip contentStyle={tooltipStyle} formatter={(v) => `${v.toFixed(2)}%`} />
              <Bar dataKey="fpr" radius={[3, 3, 0, 0]}>
                {diagnostics.map((d) => (
                  <Cell key={d.short} fill={d.winner ? '#10b981' : '#ef4444'} fillOpacity={d.winner ? 1 : 0.45} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Model-by-model interactive filter */}
      <div className="glass-panel p-5 mb-6">
        <h2 className="text-sm font-bold uppercase tracking-widest mb-4" style={{ color: '#94a3b8' }}>
          Inspect Model Diagnostics
        </h2>
        <div className="flex flex-wrap gap-2 mb-5">
          {diagnostics.map((d) => (
            <button
              key={d.short}
              onClick={() => setSelected(d.short)}
              className="px-3 py-1.5 rounded text-xs font-bold uppercase tracking-wider transition-colors"
              style={{
                background: selected === d.short ? (d.winner ? 'rgba(16,185,129,0.15)' : 'rgba(6,182,212,0.15)') : 'rgba(0,0,0,0.3)',
                border: `1px solid ${selected === d.short ? (d.winner ? '#10b981' : '#06b6d4') : '#1e293b'}`,
                color: selected === d.short ? (d.winner ? '#10b981' : '#06b6d4') : '#64748b',
              }}
            >
              {d.name} {d.winner && '🏆'}
            </button>
          ))}
        </div>

        <div className="flex flex-wrap gap-4">
          <StatCard
            label="Boundary IoU (4px)"
            value={active.boundaryIoU.toFixed(4)}
            color="var(--color-tactical-cyan)"
            icon={Crosshair}
            sub={active.short === 'Base' ? 'Reference' : `${active.boundaryIoU > baseline.boundaryIoU ? '+' : ''}${((active.boundaryIoU - baseline.boundaryIoU) / baseline.boundaryIoU * 100).toFixed(1)}% vs Baseline`}
          />
          <StatCard
            label="False Positive Rate"
            value={active.fpr.toFixed(2)}
            unit="%"
            color="var(--color-tactical-crimson)"
            icon={ShieldOff}
            sub={active.short === 'Base' ? 'Reference' : `${active.fpr > expM.fpr ? '+' : ''}${((active.fpr - expM.fpr) / expM.fpr * 100).toFixed(1)}% vs Exp M`}
          />
          <StatCard
            label="Background Leakage"
            value={active.bgLeakage.toFixed(2)}
            unit="%"
            color="var(--color-tactical-amber)"
            icon={active.bgLeakage <= expM.bgLeakage ? TrendingDown : TrendingUp}
          />
        </div>
      </div>

      {/* Visual Evidence Gallery */}
      <div className="glass-panel p-5 mb-6">
        <h2 className="flex items-center gap-2 text-sm font-bold uppercase tracking-widest mb-4" style={{ color: '#94a3b8' }}>
          <ImageIcon className="w-4 h-4" /> Visual Evidence Gallery
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {gallery.map((item) => (
            <button
              key={item.file}
              onClick={() => !failedImages[item.file] && setLightbox(item)}
              className="text-left rounded overflow-hidden group"
              style={{ background: '#05080c', border: '1px solid #1e293b' }}
            >
              <div className="relative overflow-hidden" style={{ aspectRatio: '16/9', background: '#0a0d12' }}>
                {failedImages[item.file] ? (
                  <div className="w-full h-full flex flex-col items-center justify-center gap-2" style={{ color: '#334155' }}>
                    <ImageIcon className="w-8 h-8" />
                    <span className="text-xs uppercase tracking-wide px-4 text-center">
                      Asset unavailable — start the backend server on :5000
                    </span>
                  </div>
                ) : (
                  <img
                    src={`${API_BASE}/artifacts/${item.file}`}
                    alt={item.title}
                    className="w-full h-full object-cover transition-transform duration-300 group-hover:scale-105"
                    onError={() => setFailedImages((prev) => ({ ...prev, [item.file]: true }))}
                  />
                )}
              </div>
              <div className="p-3">
                <p className="text-sm font-bold" style={{ color: '#e2e8f0' }}>{item.title}</p>
                <p className="text-xs mt-1 leading-relaxed" style={{ color: '#64748b' }}>{item.caption}</p>
              </div>
            </button>
          ))}
        </div>
      </div>

      {/* Lightbox */}
      {lightbox && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center p-6"
          style={{ background: 'rgba(2,4,8,0.92)', backdropFilter: 'blur(4px)' }}
          onClick={() => setLightbox(null)}
        >
          <div className="max-w-5xl w-full" onClick={(e) => e.stopPropagation()}>
            <div className="flex justify-between items-center mb-3">
              <p className="text-sm font-bold uppercase tracking-wider" style={{ color: 'var(--color-tactical-cyan)' }}>{lightbox.title}</p>
              <button onClick={() => setLightbox(null)} style={{ color: '#94a3b8' }}>
                <X className="w-5 h-5" />
              </button>
            </div>
            <img
              src={`${API_BASE}/artifacts/${lightbox.file}`}
              alt={lightbox.title}
              className="w-full rounded"
              style={{ border: '1px solid #1e293b' }}
            />
          </div>
        </div>
      )}
    </div>
  );
}
