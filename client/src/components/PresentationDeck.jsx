import React, { useState, useEffect, useCallback } from 'react';
import {
  ChevronLeft, ChevronRight, Crosshair, Eye, Layers, Database,
  Award, ShieldOff, Rocket, Target
} from 'lucide-react';

const experiments = [
  { name: 'Baseline', sAlpha: 0.7647, ePhi: 0.8061, fBetaW: 0.6036, mae: 0.0593, dataset: 'Off-the-shelf COD', pairs: 'N/A' },
  { name: 'Exp M',    sAlpha: 0.8638, ePhi: 0.9046, fBetaW: 0.7718, mae: 0.0348, dataset: 'Military',          pairs: '1,498' },
  { name: 'Exp C1',   sAlpha: 0.7572, ePhi: 0.8045, fBetaW: 0.5886, mae: 0.0538, dataset: 'CAMO',              pairs: '1,000' },
  { name: 'Exp C2',   sAlpha: 0.6223, ePhi: 0.6616, fBetaW: 0.3541, mae: 0.1007, dataset: 'COD10K',            pairs: '6,000' },
  { name: 'Exp MC1',  sAlpha: 0.8675, ePhi: 0.9095, fBetaW: 0.7773, mae: 0.0322, dataset: 'Military + CAMO',   pairs: '2,498', winner: true },
  { name: 'Exp MC2',  sAlpha: 0.8460, ePhi: 0.8765, fBetaW: 0.7276, mae: 0.0423, dataset: 'Military + COD10K', pairs: '7,498' },
  { name: 'Exp MCC',  sAlpha: 0.8518, ePhi: 0.8802, fBetaW: 0.7379, mae: 0.0408, dataset: 'Mil + CAMO + COD',  pairs: '8,498' },
];

function SlideShell({ eyebrow, title, icon: Icon, children }) {
  return (
    <div className="w-full h-full flex flex-col p-10 md:p-14">
      <div className="flex items-center gap-3 mb-2">
        {Icon && <Icon className="w-5 h-5" style={{ color: 'var(--color-tactical-cyan)' }} />}
        <span className="text-xs font-bold uppercase tracking-[0.2em]" style={{ color: 'var(--color-tactical-cyan)' }}>{eyebrow}</span>
      </div>
      <h1 className="text-2xl md:text-4xl font-bold uppercase tracking-wide mb-8" style={{ color: '#f8fafc' }}>{title}</h1>
      <div className="flex-1 min-h-0">{children}</div>
    </div>
  );
}

function Bullet({ children, color = '#10b981' }) {
  return (
    <li className="flex gap-3 items-start text-sm md:text-base leading-relaxed" style={{ color: '#cbd5e1' }}>
      <span className="mt-2 w-1.5 h-1.5 rounded-full flex-shrink-0" style={{ background: color }} />
      <span>{children}</span>
    </li>
  );
}

const slides = [
  // 1. Title
  {
    render: () => (
      <div className="w-full h-full flex flex-col items-center justify-center text-center p-10 relative">
        <div className="absolute inset-0 opacity-30" style={{
          backgroundImage: 'radial-gradient(circle at center, rgba(16,185,129,0.15) 0%, transparent 60%)'
        }} />
        <Target className="w-14 h-14 mb-6" style={{ color: 'var(--color-tactical-emerald)' }} />
        <p className="text-xs font-bold uppercase tracking-[0.3em] mb-4" style={{ color: 'var(--color-tactical-cyan)' }}>
          Ghost-Recon-CV // Tactical Intelligence Platform
        </p>
        <h1 className="text-3xl md:text-5xl font-bold uppercase tracking-wide mb-4 max-w-4xl" style={{ color: '#f8fafc' }}>
          Camouflaged Object Detection<br />in Tactical Settings
        </h1>
        <p className="text-sm md:text-base max-w-2xl mb-6" style={{ color: '#94a3b8' }}>
          Replicating &amp; extending Table 6 of Hwang &amp; Ma (2024) with SINet-V2, evaluated against the
          constraints of the military <strong style={{ color: 'var(--color-tactical-amber)' }}>OODA loop</strong> —
          Observe, Orient, Decide, Act.
        </p>
        <div className="flex gap-6 text-xs uppercase tracking-widest" style={{ color: '#475569' }}>
          <span>SINet-V2 · Res2Net-50</span>
          <span>·</span>
          <span>580-Image Military Test Set</span>
          <span>·</span>
          <span>7 Trained Regimes</span>
        </div>
      </div>
    ),
  },
  // 2. The Military Camouflage Problem
  {
    render: () => (
      <SlideShell eyebrow="Operational Context" title="The Military Camouflage Problem" icon={ShieldOff}>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8 h-full">
          <div>
            <p className="text-sm md:text-base leading-relaxed mb-5" style={{ color: '#cbd5e1' }}>
              Modern camouflage detection models are trained almost exclusively on <em>natural</em> camouflage —
              animals and insects blending into foliage. Military uniforms and ghillie suits are engineered by
              humans with a fundamentally different visual statistic, creating a severe <strong style={{ color: 'var(--color-tactical-crimson)' }}>domain gap</strong>.
            </p>
            <ul className="space-y-3">
              <Bullet color="#ef4444">Off-the-shelf SINet-V2 scores just <strong>F<sub>β</sub><sup>w</sup> = 0.6036</strong> on unseen tactical personnel.</Bullet>
              <Bullet color="#ef4444">Training solely on diverse wildlife (COD10K) causes <strong>negative transfer</strong> — F<sub>β</sub><sup>w</sup> collapses to 0.3541.</Bullet>
              <Bullet color="#f59e0b">In the OODA loop, missed detections are lethal; false alarms cause cognitive overload during "Orient".</Bullet>
            </ul>
          </div>
          <div className="rounded-lg p-6 flex flex-col justify-center" style={{ background: 'rgba(239,68,68,0.06)', border: '1px solid rgba(239,68,68,0.25)' }}>
            <p className="text-xs uppercase tracking-widest font-bold mb-2" style={{ color: 'var(--color-tactical-crimson)' }}>Negative Transfer, Quantified</p>
            <p className="font-mono text-4xl font-bold mb-1" style={{ color: 'var(--color-tactical-crimson)' }}>0.3541</p>
            <p className="text-xs" style={{ color: '#94a3b8' }}>Exp C2 (COD10K-only) weighted F-measure — a 41.3% relative drop from Baseline, despite 6,000 training pairs.</p>
          </div>
        </div>
      </SlideShell>
    ),
  },
  // 3. Architecture
  {
    render: () => (
      <SlideShell eyebrow="Methodology" title="Architecture — SINet-V2" icon={Layers}>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-5 h-full">
          {[
            { title: 'Res2Net-50 Backbone', desc: 'Multi-scale feature extraction with hierarchical residual-like connections within a single block, capturing fine-grained texture at multiple receptive fields.', color: '#06b6d4' },
            { title: 'Neighbor Connection Decoder', desc: 'Aggregates high-level semantic features for coarse search & localization — the "where is it" stage of camouflaged detection.', color: '#f59e0b' },
            { title: 'Group Reversal Attention', desc: 'Iteratively refines boundaries via reverse attention across groups, sharpening silhouette contours critical for soldier identification.', color: '#10b981' },
          ].map((s) => (
            <div key={s.title} className="rounded-lg p-5 flex flex-col" style={{ background: 'rgba(0,0,0,0.3)', border: `1px solid ${s.color}44` }}>
              <p className="text-sm font-bold uppercase tracking-wide mb-3" style={{ color: s.color }}>{s.title}</p>
              <p className="text-xs md:text-sm leading-relaxed" style={{ color: '#94a3b8' }}>{s.desc}</p>
            </div>
          ))}
        </div>
        <p className="text-xs mt-6 uppercase tracking-widest" style={{ color: '#475569' }}>
          Search-and-Identify pipeline mimics human visual perception when actively hunting for hidden threats.
        </p>
      </SlideShell>
    ),
  },
  // 4. Dataset Architecture
  {
    render: () => (
      <SlideShell eyebrow="Data Foundation" title="Dataset Architecture" icon={Database}>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
          {[
            { name: 'ACD1K + Military Personnel', train: '1,498', test: '580', desc: 'Artificial tactical camouflage — uniforms, ghillie suits, field gear.', color: '#10b981' },
            { name: 'CAMO', train: '1,000', test: '250', desc: 'Natural camouflage — animals in natural habitats.', color: '#06b6d4' },
            { name: 'COD10K', train: '6,000', test: '4,000', desc: 'Large-scale, highly diverse natural camouflage.', color: '#f59e0b' },
          ].map((d) => (
            <div key={d.name} className="rounded-lg p-5" style={{ background: 'rgba(0,0,0,0.3)', border: '1px solid #1e293b' }}>
              <p className="text-sm font-bold mb-3" style={{ color: d.color }}>{d.name}</p>
              <div className="flex gap-4 mb-3">
                <div>
                  <p className="font-mono text-2xl font-bold" style={{ color: '#e2e8f0' }}>{d.train}</p>
                  <p className="text-xs uppercase tracking-widest" style={{ color: '#475569' }}>Train Pairs</p>
                </div>
                <div>
                  <p className="font-mono text-2xl font-bold" style={{ color: '#e2e8f0' }}>{d.test}</p>
                  <p className="text-xs uppercase tracking-widest" style={{ color: '#475569' }}>Test Pairs</p>
                </div>
              </div>
              <p className="text-xs leading-relaxed" style={{ color: '#94a3b8' }}>{d.desc}</p>
            </div>
          ))}
        </div>
        <p className="text-xs mt-6" style={{ color: '#64748b' }}>
          All 7 experiments are evaluated on the same standardized <strong style={{ color: '#e2e8f0' }}>580-image Military Test split</strong> for direct comparability.
        </p>
      </SlideShell>
    ),
  },
  // 5. Table 6 Results
  {
    render: () => (
      <SlideShell eyebrow="Ablation Study" title="Table 6 — Combination Results" icon={Award}>
        <div className="overflow-x-auto">
          <table className="w-full text-xs md:text-sm">
            <thead>
              <tr style={{ borderBottom: '1px solid #1e293b', color: '#64748b' }}>
                <th className="text-left py-2 pr-4 uppercase tracking-wide">Experiment</th>
                <th className="text-left py-2 pr-4 uppercase tracking-wide">Training Data</th>
                <th className="text-center py-2 px-2 uppercase tracking-wide">S<sub>α</sub></th>
                <th className="text-center py-2 px-2 uppercase tracking-wide">E<sub>φ</sub></th>
                <th className="text-center py-2 px-2 uppercase tracking-wide">F<sub>β</sub><sup>w</sup></th>
                <th className="text-center py-2 px-2 uppercase tracking-wide">MAE</th>
              </tr>
            </thead>
            <tbody>
              {experiments.map((e) => (
                <tr key={e.name} style={{ borderBottom: '1px solid #0f172a', background: e.winner ? 'rgba(16,185,129,0.08)' : 'transparent' }}>
                  <td className="py-2 pr-4 font-mono font-bold" style={{ color: e.winner ? 'var(--color-tactical-emerald)' : '#e2e8f0' }}>
                    {e.name} {e.winner && '🏆'}
                  </td>
                  <td className="py-2 pr-4" style={{ color: '#64748b' }}>{e.dataset}</td>
                  <td className="py-2 px-2 text-center font-mono" style={{ color: '#cbd5e1' }}>{e.sAlpha.toFixed(4)}</td>
                  <td className="py-2 px-2 text-center font-mono" style={{ color: '#cbd5e1' }}>{e.ePhi.toFixed(4)}</td>
                  <td className="py-2 px-2 text-center font-mono font-bold" style={{ color: e.winner ? 'var(--color-tactical-emerald)' : '#cbd5e1' }}>{e.fBetaW.toFixed(4)}</td>
                  <td className="py-2 px-2 text-center font-mono" style={{ color: '#cbd5e1' }}>{e.mae.toFixed(4)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="mt-6 rounded-lg p-4 flex items-center gap-3" style={{ background: 'rgba(16,185,129,0.08)', border: '1px solid rgba(16,185,129,0.3)' }}>
          <Award className="w-6 h-6 flex-shrink-0" style={{ color: 'var(--color-tactical-emerald)' }} />
          <p className="text-xs md:text-sm" style={{ color: '#cbd5e1' }}>
            <strong style={{ color: 'var(--color-tactical-emerald)' }}>Exp MC1 (Military + CAMO)</strong> is the undisputed optimum
            across every metric — fusing natural and artificial camouflage domains produces the best regularization.
          </p>
        </div>
      </SlideShell>
    ),
  },
  // 6. Contour Sharpness & False Alarm Suppression
  {
    render: () => (
      <SlideShell eyebrow="Qualitative Diagnostics" title="Contour Sharpness & False Alarm Suppression" icon={Crosshair}>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 h-full">
          <div className="rounded-lg p-6" style={{ background: 'rgba(6,182,212,0.06)', border: '1px solid rgba(6,182,212,0.25)' }}>
            <p className="text-xs uppercase tracking-widest font-bold mb-2" style={{ color: 'var(--color-tactical-cyan)' }}>Boundary IoU (4px band)</p>
            <p className="font-mono text-4xl font-bold mb-2" style={{ color: 'var(--color-tactical-cyan)' }}>0.3437</p>
            <p className="text-xs md:text-sm leading-relaxed" style={{ color: '#94a3b8' }}>
              Exp MC1 achieves the sharpest soldier silhouettes of any regime — a +47.4% relative gain over Baseline
              (0.2332) and ahead of military-only training (0.3372).
            </p>
          </div>
          <div className="rounded-lg p-6" style={{ background: 'rgba(16,185,129,0.06)', border: '1px solid rgba(16,185,129,0.25)' }}>
            <p className="text-xs uppercase tracking-widest font-bold mb-2" style={{ color: 'var(--color-tactical-emerald)' }}>False Alarm Reduction</p>
            <p className="font-mono text-4xl font-bold mb-2" style={{ color: 'var(--color-tactical-emerald)' }}>−23%</p>
            <p className="text-xs md:text-sm leading-relaxed" style={{ color: '#94a3b8' }}>
              FPR drops from 2.02% (Exp M) to 1.55% (Exp MC1) on tactical background foliage and terrain clutter —
              fewer false triggers during the OODA loop's "Orient" phase.
            </p>
          </div>
        </div>
      </SlideShell>
    ),
  },
  // 7. Future Work
  {
    render: () => (
      <SlideShell eyebrow="Path Forward" title="Tactical C2 Integration & Future Work" icon={Rocket}>
        <ul className="space-y-4">
          <Bullet color="#06b6d4"><strong style={{ color: '#e2e8f0' }}>Edge AI Deployment</strong> — quantize SINet-V2 for real-time inference on UAV-mounted edge hardware (Jetson-class), targeting &gt;15 FPS at 352×352.</Bullet>
          <Bullet color="#f59e0b"><strong style={{ color: '#e2e8f0' }}>Command & Control Dashboards</strong> — the Ghost-Recon-CV platform built here (React + FastAPI sidecar) is the first step toward a live tactical HUD.</Bullet>
          <Bullet color="#10b981"><strong style={{ color: '#e2e8f0' }}>Multi-Domain Curriculum</strong> — extend the MC1 fusion insight to thermal and night-vision modalities for 24-hour observation coverage.</Bullet>
          <Bullet color="#ef4444"><strong style={{ color: '#e2e8f0' }}>Soldier Survivability</strong> — sharper boundaries and lower false-alarm rates directly compress OODA loop latency, the core determinant of survivability in contact.</Bullet>
        </ul>
        <div className="mt-8 flex items-center gap-3 text-xs uppercase tracking-widest" style={{ color: '#475569' }}>
          <Eye className="w-4 h-4" /> Ghost-Recon-CV — Hwang &amp; Ma (2024) Replication &amp; Extension
        </div>
      </SlideShell>
    ),
  },
];

export default function PresentationDeck() {
  const [index, setIndex] = useState(0);
  const total = slides.length;

  const next = useCallback(() => setIndex((i) => Math.min(i + 1, total - 1)), [total]);
  const prev = useCallback(() => setIndex((i) => Math.max(i - 1, 0)), []);

  useEffect(() => {
    const onKey = (e) => {
      if (e.key === 'ArrowRight') next();
      if (e.key === 'ArrowLeft') prev();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [next, prev]);

  return (
    <div className="p-6 max-w-6xl mx-auto flex flex-col" style={{ height: 'calc(100vh - 120px)' }}>
      <div className="glass-panel flex-1 min-h-0 relative overflow-hidden">
        <div className="scanline" />
        {slides[index].render()}
      </div>

      {/* Controls */}
      <div className="glass-panel mt-4 p-4 flex items-center gap-4">
        <button
          onClick={prev}
          disabled={index === 0}
          className="p-2 rounded transition-colors disabled:opacity-30"
          style={{ background: '#1e293b', border: '1px solid #334155', color: '#e2e8f0' }}
        >
          <ChevronLeft className="w-4 h-4" />
        </button>

        <div className="flex-1">
          <div className="flex justify-between text-xs uppercase tracking-widest mb-1.5" style={{ color: '#64748b' }}>
            <span>Slide {index + 1} of {total}</span>
            <span style={{ color: '#334155' }}>← → to navigate</span>
          </div>
          <div className="w-full h-1.5 rounded-full" style={{ background: '#1e293b' }}>
            <div
              className="h-full rounded-full transition-all duration-300"
              style={{ width: `${((index + 1) / total) * 100}%`, background: 'var(--color-tactical-emerald)' }}
            />
          </div>
        </div>

        <button
          onClick={next}
          disabled={index === total - 1}
          className="p-2 rounded transition-colors disabled:opacity-30"
          style={{ background: '#1e293b', border: '1px solid #334155', color: '#e2e8f0' }}
        >
          <ChevronRight className="w-4 h-4" />
        </button>
      </div>
    </div>
  );
}
