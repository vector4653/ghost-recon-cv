import React, { useState } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell
} from 'recharts';
import { ImageIcon, X } from 'lucide-react';
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

const tooltipStyle = { backgroundColor: '#11161d', borderColor: '#2b3542', borderRadius: 6, color: '#e5e7eb', fontSize: 12 };
const axisTick = { fill: '#64748b', fontSize: 11 };

function StatCard({ label, value, unit, sub }) {
  return (
    <div className="rounded-md p-4 flex-1 min-w-[160px]" style={{ background: 'var(--bg)', border: '1px solid var(--border)' }}>
      <p className="text-xs mb-1" style={{ color: 'var(--text-muted)' }}>{label}</p>
      <p className="num text-2xl">{value}<span className="text-sm ml-0.5" style={{ color: 'var(--text-faint)' }}>{unit}</span></p>
      {sub && <p className="text-xs mt-1" style={{ color: 'var(--text-faint)' }}>{sub}</p>}
    </div>
  );
}

function MetricChart({ dataKey, domain, unit, format }) {
  return (
    <ResponsiveContainer width="100%" height={200}>
      <BarChart data={diagnostics} margin={{ top: 5, right: 10, left: -20, bottom: 0 }}>
        <CartesianGrid stroke="#1f2733" vertical={false} />
        <XAxis dataKey="short" stroke="#2b3542" tick={axisTick} tickLine={false} />
        <YAxis stroke="#2b3542" tick={axisTick} tickLine={false} axisLine={false} domain={domain} unit={unit} />
        <Tooltip contentStyle={tooltipStyle} cursor={{ fill: 'rgba(255,255,255,0.03)' }} formatter={format} />
        <Bar isAnimationActive={false} dataKey={dataKey} radius={[3, 3, 0, 0]}>
          {diagnostics.map((d) => (
            <Cell key={d.short} fill={d.winner ? '#10b981' : '#475569'} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

export default function DiagnosticsViewer() {
  const [selected, setSelected] = useState(winnerEntry.short);
  const [lightbox, setLightbox] = useState(null);
  const [failedImages, setFailedImages] = useState({});
  const active = diagnostics.find(d => d.short === selected);

  return (
    <div className="flex flex-col gap-6">
      <div>
        <h1 className="text-xl font-semibold tracking-tight">Diagnostics</h1>
        <p className="text-sm mt-1" style={{ color: 'var(--text-muted)' }}>
          Boundary precision, false alarms and visual comparisons across all models.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <section className="panel p-4">
          <h2 className="panel-title">Boundary sharpness (4px contour IoU)</h2>
          <p className="text-xs leading-relaxed mt-1 mb-4" style={{ color: 'var(--text-muted)' }}>
            IoU measured only within a 4px band around each object outline. Exp MC1 leads at 0.3437,
            +47% over the Baseline (0.2332).
          </p>
          <MetricChart dataKey="boundaryIoU" domain={[0, 0.4]} format={(v) => v.toFixed(4)} />
        </section>

        <section className="panel p-4">
          <h2 className="panel-title">False positive rate</h2>
          <p className="text-xs leading-relaxed mt-1 mb-4" style={{ color: 'var(--text-muted)' }}>
            Detections on background foliage and clutter. Adding CAMO data cuts FPR to 1.55% for Exp MC1,
            about 23% lower than military-only training (Exp M, 2.02%).
          </p>
          <MetricChart dataKey="fpr" domain={[0, 2.5]} unit="%" format={(v) => `${v.toFixed(2)}%`} />
        </section>
      </div>

      <section className="panel p-4">
        <h2 className="panel-title mb-3">Compare models</h2>
        <div className="flex flex-wrap gap-1.5 mb-4">
          {diagnostics.map((d) => (
            <button
              key={d.short}
              onClick={() => setSelected(d.short)}
              className="chip"
              aria-pressed={selected === d.short}
            >
              {d.name}
            </button>
          ))}
        </div>

        <div className="flex flex-wrap gap-3">
          <StatCard
            label="Boundary IoU (4px)"
            value={active.boundaryIoU.toFixed(4)}
            sub={active.short === 'Base' ? 'Reference' : `${active.boundaryIoU > baseline.boundaryIoU ? '+' : ''}${((active.boundaryIoU - baseline.boundaryIoU) / baseline.boundaryIoU * 100).toFixed(1)}% vs Baseline`}
          />
          <StatCard
            label="False positive rate"
            value={active.fpr.toFixed(2)}
            unit="%"
            sub={active.short === 'Base' ? 'Reference' : `${active.fpr > expM.fpr ? '+' : ''}${((active.fpr - expM.fpr) / expM.fpr * 100).toFixed(1)}% vs Exp M`}
          />
          <StatCard
            label="Background leakage"
            value={active.bgLeakage.toFixed(2)}
            unit="%"
          />
        </div>
      </section>

      <section className="panel p-4">
        <h2 className="panel-title mb-4">Visual comparisons</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {gallery.map((item) => (
            <button
              key={item.file}
              onClick={() => !failedImages[item.file] && setLightbox(item)}
              className="text-left rounded-md overflow-hidden"
              style={{ background: 'var(--bg)', border: '1px solid var(--border)' }}
            >
              <div className="relative overflow-hidden" style={{ aspectRatio: '16/9' }}>
                {failedImages[item.file] ? (
                  <div className="w-full h-full flex flex-col items-center justify-center gap-2" style={{ color: 'var(--text-faint)' }}>
                    <ImageIcon className="w-8 h-8" strokeWidth={1.5} />
                    <span className="text-xs px-4 text-center">Image unavailable. Is the backend server running?</span>
                  </div>
                ) : (
                  <img
                    src={`${API_BASE}/artifacts/${item.file}`}
                    alt={item.title}
                    className="w-full h-full object-contain" style={{ background: "#fff" }}
                    onError={() => setFailedImages((prev) => ({ ...prev, [item.file]: true }))}
                  />
                )}
              </div>
              <div className="p-3">
                <p className="text-sm font-medium">{item.title}</p>
                <p className="text-xs mt-1 leading-relaxed" style={{ color: 'var(--text-muted)' }}>{item.caption}</p>
              </div>
            </button>
          ))}
        </div>
      </section>

      {lightbox && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center p-6"
          style={{ background: 'rgba(5,8,12,0.9)' }}
          onClick={() => setLightbox(null)}
        >
          <div className="max-w-5xl w-full" onClick={(e) => e.stopPropagation()}>
            <div className="flex justify-between items-center mb-3">
              <p className="text-sm font-medium">{lightbox.title}</p>
              <button onClick={() => setLightbox(null)} style={{ color: 'var(--text-muted)' }} aria-label="Close">
                <X className="w-5 h-5" />
              </button>
            </div>
            <img
              src={`${API_BASE}/artifacts/${lightbox.file}`}
              alt={lightbox.title}
              className="w-full rounded-md"
            />
          </div>
        </div>
      )}
    </div>
  );
}
