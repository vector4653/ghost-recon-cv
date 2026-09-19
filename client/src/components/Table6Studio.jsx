import React, { useMemo, useState } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis
} from 'recharts';
import { TrendingUp, Award, ChevronUp, ChevronDown, ArrowUpDown } from 'lucide-react';

const experiments = [
  { name: 'Baseline', short: 'Base', sAlpha: 0.7647, ePhi: 0.8061, fBetaW: 0.6036, mae: 0.0593, dataset: 'Off-the-shelf COD', pairs: 'N/A', winner: false },
  { name: 'Exp M',    short: 'M',    sAlpha: 0.8638, ePhi: 0.9046, fBetaW: 0.7718, mae: 0.0348, dataset: 'Military',          pairs: '1,498', winner: false },
  { name: 'Exp C1',   short: 'C1',   sAlpha: 0.7572, ePhi: 0.8045, fBetaW: 0.5886, mae: 0.0538, dataset: 'CAMO',              pairs: '1,000', winner: false },
  { name: 'Exp C2',   short: 'C2',   sAlpha: 0.6223, ePhi: 0.6616, fBetaW: 0.3541, mae: 0.1007, dataset: 'COD10K',            pairs: '6,000', winner: false },
  { name: 'Exp MC1',  short: 'MC1',  sAlpha: 0.8675, ePhi: 0.9095, fBetaW: 0.7773, mae: 0.0322, dataset: 'Military + CAMO',   pairs: '2,498', winner: true  },
  { name: 'Exp MC2',  short: 'MC2',  sAlpha: 0.8460, ePhi: 0.8765, fBetaW: 0.7276, mae: 0.0423, dataset: 'Military + COD10K', pairs: '7,498', winner: false },
  { name: 'Exp MCC',  short: 'MCC',  sAlpha: 0.8518, ePhi: 0.8802, fBetaW: 0.7379, mae: 0.0408, dataset: 'Mil + CAMO + COD',  pairs: '8,498', winner: false },
];

const baseline = experiments[0];

function Delta({ val, base, higherIsBetter = true }) {
  const diff = (val - base).toFixed(4);
  const better = higherIsBetter ? diff > 0 : diff < 0;
  const neutral = Math.abs(diff) < 0.0001;
  if (neutral) return <span style={{ color: '#475569' }}>—</span>;
  return (
    <span className="flex items-center gap-0.5 text-xs" style={{ color: better ? '#10b981' : '#ef4444' }}>
      {better ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
      {Math.abs(diff)}
    </span>
  );
}

const tooltipStyle = { backgroundColor: '#0a0d12', borderColor: '#1e293b', color: '#f8fafc', fontSize: 12 };

const columns = [
  { key: 'name', label: 'Experiment', align: 'left', higherIsBetter: true },
  { key: 'sAlpha', label: <>S<sub>α</sub> ↑</>, align: 'center', higherIsBetter: true },
  { key: 'ePhi', label: <>E<sub>φ</sub> ↑</>, align: 'center', higherIsBetter: true },
  { key: 'fBetaW', label: <>F<sub>β</sub><sup>w</sup> ↑</>, align: 'center', higherIsBetter: true },
  { key: 'mae', label: 'MAE ↓', align: 'center', higherIsBetter: false },
];

function SortHeader({ col, sortKey, sortDir, onSort }) {
  const isActive = sortKey === col.key;
  return (
    <th
      onClick={() => onSort(col.key)}
      className={`py-2 ${col.align === 'left' ? 'text-left pr-4' : 'text-center px-3'} font-medium uppercase tracking-wide text-xs select-none cursor-pointer`}
      style={{ color: isActive ? 'var(--color-tactical-cyan)' : '#64748b' }}
    >
      <span className={`inline-flex items-center gap-1 ${col.align === 'center' ? 'justify-center' : ''}`}>
        {col.label}
        {isActive ? (
          sortDir === 'asc' ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />
        ) : (
          <ArrowUpDown className="w-3 h-3 opacity-30" />
        )}
      </span>
    </th>
  );
}

export default function Table6Studio() {
  const [sortKey, setSortKey] = useState(null);
  const [sortDir, setSortDir] = useState('desc');

  const handleSort = (key) => {
    if (sortKey === key) {
      setSortDir((d) => (d === 'asc' ? 'desc' : 'asc'));
    } else {
      setSortKey(key);
      const col = columns.find((c) => c.key === key);
      setSortDir(col?.higherIsBetter === false ? 'asc' : 'desc');
    }
  };

  const sortedExperiments = useMemo(() => {
    if (!sortKey) return experiments;
    const rows = [...experiments];
    rows.sort((a, b) => {
      const av = a[sortKey];
      const bv = b[sortKey];
      if (typeof av === 'string') return sortDir === 'asc' ? av.localeCompare(bv) : bv.localeCompare(av);
      return sortDir === 'asc' ? av - bv : bv - av;
    });
    return rows;
  }, [sortKey, sortDir]);

  return (
    <div className="p-6 max-w-7xl mx-auto overflow-y-auto" style={{ maxHeight: 'calc(100vh - 120px)' }}>

      {/* Header */}
      <div className="glass-panel p-5 mb-6 flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div>
          <h1 className="text-xl font-bold uppercase tracking-wider mb-1" style={{ color: 'var(--color-tactical-amber)' }}>
            Table 6 — Ablation Analysis
          </h1>
          <p className="text-sm" style={{ color: '#64748b' }}>
            Multi-domain camouflage transfer evaluated on Military Test split (580 images)
          </p>
        </div>
        <div className="flex items-center gap-3 rounded-lg p-3" style={{ background: 'rgba(16,185,129,0.1)', border: '1px solid rgba(16,185,129,0.3)' }}>
          <Award className="w-7 h-7" style={{ color: 'var(--color-tactical-emerald)' }} />
          <div>
            <p className="text-xs uppercase tracking-widest font-bold" style={{ color: 'var(--color-tactical-emerald)' }}>Optimal Regime</p>
            <p className="text-base font-bold text-white">Exp MC1 — Military + CAMO</p>
          </div>
        </div>
      </div>

      {/* Metric Table */}
      <div className="glass-panel p-5 mb-6 overflow-x-auto">
        <h2 className="text-sm font-bold uppercase tracking-widest mb-4" style={{ color: '#94a3b8' }}>
          Full Metric Matrix — All 7 Experiments
        </h2>
        <p className="text-xs mb-2" style={{ color: '#475569' }}>Click a column header to sort.</p>
        <table className="w-full text-sm">
          <thead>
            <tr style={{ borderBottom: '1px solid #1e293b', color: '#64748b' }}>
              {columns.map((col, i) => (
                i === 0 ? (
                  <React.Fragment key={col.key}>
                    <SortHeader col={col} sortKey={sortKey} sortDir={sortDir} onSort={handleSort} />
                    <th className="text-left py-2 pr-4 font-medium uppercase tracking-wide text-xs">Training Data</th>
                  </React.Fragment>
                ) : (
                  <SortHeader key={col.key} col={col} sortKey={sortKey} sortDir={sortDir} onSort={handleSort} />
                )
              ))}
              <th className="text-center py-2 px-3 font-medium uppercase tracking-wide text-xs">ΔF<sub>β</sub><sup>w</sup></th>
            </tr>
          </thead>
          <tbody>
            {sortedExperiments.map((exp) => (
              <tr
                key={exp.name}
                style={{
                  borderBottom: '1px solid #0f172a',
                  background: exp.winner ? 'rgba(16,185,129,0.06)' : 'transparent',
                }}
              >
                <td className="py-2.5 pr-4 font-mono font-bold" style={{ color: exp.winner ? 'var(--color-tactical-emerald)' : '#e2e8f0' }}>
                  {exp.name} {exp.winner && '🏆'}
                </td>
                <td className="py-2.5 pr-4 text-xs" style={{ color: '#64748b' }}>{exp.dataset}</td>
                <td className="py-2.5 px-3 text-center font-mono" style={{ color: '#cbd5e1' }}>{exp.sAlpha.toFixed(4)}</td>
                <td className="py-2.5 px-3 text-center font-mono" style={{ color: '#cbd5e1' }}>{exp.ePhi.toFixed(4)}</td>
                <td className="py-2.5 px-3 text-center font-mono font-bold" style={{ color: exp.winner ? 'var(--color-tactical-emerald)' : '#cbd5e1' }}>{exp.fBetaW.toFixed(4)}</td>
                <td className="py-2.5 px-3 text-center font-mono" style={{ color: exp.mae > 0.08 ? 'var(--color-tactical-crimson)' : '#cbd5e1' }}>{exp.mae.toFixed(4)}</td>
                <td className="py-2.5 px-3 flex justify-center">
                  <Delta val={exp.fBetaW} base={baseline.fBetaW} />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">

        {/* Bar Chart — S-alpha & E-phi */}
        <div className="glass-panel p-5" style={{ height: 380 }}>
          <h2 className="text-xs font-bold uppercase tracking-widest mb-4 flex items-center gap-2" style={{ color: '#94a3b8' }}>
            <TrendingUp className="w-4 h-4" style={{ color: 'var(--color-tactical-cyan)' }} />
            Structure & Alignment Metrics (S<sub>α</sub>, E<sub>φ</sub>)
          </h2>
          <ResponsiveContainer width="100%" height="88%">
            <BarChart data={experiments} margin={{ top: 5, right: 10, left: -25, bottom: 25 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
              <XAxis dataKey="short" stroke="#334155" tick={{ fill: '#64748b', fontSize: 11 }} />
              <YAxis stroke="#334155" tick={{ fill: '#64748b', fontSize: 11 }} domain={[0.55, 0.95]} />
              <Tooltip contentStyle={tooltipStyle} />
              <Legend wrapperStyle={{ fontSize: 12, paddingTop: 8 }} />
              <Bar dataKey="sAlpha" name="Sα (Structure)" fill="#06b6d4" radius={[3, 3, 0, 0]} />
              <Bar dataKey="ePhi"   name="Eφ (Alignment)" fill="#f59e0b"  radius={[3, 3, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Bar Chart — F-beta & MAE */}
        <div className="glass-panel p-5" style={{ height: 380 }}>
          <h2 className="text-xs font-bold uppercase tracking-widest mb-4 flex items-center gap-2" style={{ color: '#94a3b8' }}>
            <TrendingUp className="w-4 h-4" style={{ color: 'var(--color-tactical-emerald)' }} />
            Boundary & Error Metrics (F<sub>β</sub><sup>w</sup>, MAE)
          </h2>
          <ResponsiveContainer width="100%" height="88%">
            <BarChart data={experiments} margin={{ top: 5, right: 10, left: -25, bottom: 25 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
              <XAxis dataKey="short" stroke="#334155" tick={{ fill: '#64748b', fontSize: 11 }} />
              <YAxis stroke="#334155" tick={{ fill: '#64748b', fontSize: 11 }} domain={[0, 1]} />
              <Tooltip contentStyle={tooltipStyle} />
              <Legend wrapperStyle={{ fontSize: 12, paddingTop: 8 }} />
              <Bar dataKey="fBetaW" name="Fβw (F-Measure)" fill="#10b981" radius={[3, 3, 0, 0]} />
              <Bar dataKey="mae"    name="MAE (Error)"     fill="#ef4444"  radius={[3, 3, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}
