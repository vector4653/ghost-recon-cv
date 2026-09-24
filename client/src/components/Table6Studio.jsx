import React, { useMemo, useState } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer
} from 'recharts';
import { ChevronUp, ChevronDown, ArrowUpDown } from 'lucide-react';

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
  if (neutral) return <span style={{ color: 'var(--text-faint)' }}>—</span>;
  return (
    <span className="num inline-flex items-center gap-0.5 text-xs" style={{ color: better ? 'var(--color-tactical-emerald)' : 'var(--color-tactical-crimson)' }}>
      {better ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
      {Math.abs(diff)}
    </span>
  );
}

const tooltipStyle = { backgroundColor: '#11161d', borderColor: '#2b3542', borderRadius: 6, color: '#e5e7eb', fontSize: 12 };
const axisTick = { fill: '#64748b', fontSize: 11 };

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
      className={`py-2.5 ${col.align === 'left' ? 'text-left pl-4 pr-4' : 'text-center px-3'} font-medium text-xs select-none cursor-pointer`}
      style={{ color: isActive ? 'var(--text)' : 'var(--text-faint)' }}
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
    <div className="flex flex-col gap-6">
      <div className="flex flex-col sm:flex-row sm:items-end justify-between gap-2">
        <div>
          <h1 className="text-xl font-semibold tracking-tight">Ablation results</h1>
          <p className="text-sm mt-1" style={{ color: 'var(--text-muted)' }}>
            Seven training regimes evaluated on the Military test split (580 images).
          </p>
        </div>
        <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
          Best: <span style={{ color: 'var(--color-tactical-emerald)' }}>Exp MC1 (Military + CAMO)</span>
        </p>
      </div>

      <section className="panel overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr style={{ borderBottom: '1px solid var(--border)' }}>
              {columns.map((col, i) => (
                i === 0 ? (
                  <React.Fragment key={col.key}>
                    <SortHeader col={col} sortKey={sortKey} sortDir={sortDir} onSort={handleSort} />
                    <th className="text-left py-2.5 pr-4 font-medium text-xs" style={{ color: 'var(--text-faint)' }}>Training data</th>
                  </React.Fragment>
                ) : (
                  <SortHeader key={col.key} col={col} sortKey={sortKey} sortDir={sortDir} onSort={handleSort} />
                )
              ))}
              <th className="text-center py-2.5 px-3 font-medium text-xs" style={{ color: 'var(--text-faint)' }}>ΔF<sub>β</sub><sup>w</sup> vs base</th>
            </tr>
          </thead>
          <tbody>
            {sortedExperiments.map((exp) => (
              <tr
                key={exp.name}
                style={{
                  borderBottom: '1px solid var(--border)',
                  background: exp.winner ? 'rgba(16,185,129,0.06)' : 'transparent',
                }}
              >
                <td className="py-2.5 pl-4 pr-4 font-medium whitespace-nowrap" style={{ color: exp.winner ? 'var(--color-tactical-emerald)' : 'var(--text)' }}>
                  {exp.name}
                </td>
                <td className="py-2.5 pr-4 text-xs whitespace-nowrap" style={{ color: 'var(--text-muted)' }}>{exp.dataset}</td>
                <td className="py-2.5 px-3 text-center num">{exp.sAlpha.toFixed(4)}</td>
                <td className="py-2.5 px-3 text-center num">{exp.ePhi.toFixed(4)}</td>
                <td className="py-2.5 px-3 text-center num">{exp.fBetaW.toFixed(4)}</td>
                <td className="py-2.5 px-3 text-center num">{exp.mae.toFixed(4)}</td>
                <td className="py-2.5 px-3 text-center">
                  <Delta val={exp.fBetaW} base={baseline.fBetaW} />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <section className="panel p-4">
          <h2 className="panel-title mb-4">Structure &amp; alignment (S<sub>α</sub>, E<sub>φ</sub>)</h2>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={experiments} margin={{ top: 5, right: 10, left: -20, bottom: 0 }}>
              <CartesianGrid stroke="#1f2733" vertical={false} />
              <XAxis dataKey="short" stroke="#2b3542" tick={axisTick} tickLine={false} />
              <YAxis stroke="#2b3542" tick={axisTick} tickLine={false} axisLine={false} domain={[0.55, 0.95]} />
              <Tooltip contentStyle={tooltipStyle} cursor={{ fill: 'rgba(255,255,255,0.03)' }} />
              <Legend wrapperStyle={{ fontSize: 12, paddingTop: 8 }} />
              <Bar isAnimationActive={false} dataKey="sAlpha" name="Sα" fill="#10b981" radius={[3, 3, 0, 0]} />
              <Bar isAnimationActive={false} dataKey="ePhi"   name="Eφ" fill="#64748b" radius={[3, 3, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </section>

        <section className="panel p-4">
          <h2 className="panel-title mb-4">F-measure &amp; error (F<sub>β</sub><sup>w</sup>, MAE)</h2>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={experiments} margin={{ top: 5, right: 10, left: -20, bottom: 0 }}>
              <CartesianGrid stroke="#1f2733" vertical={false} />
              <XAxis dataKey="short" stroke="#2b3542" tick={axisTick} tickLine={false} />
              <YAxis stroke="#2b3542" tick={axisTick} tickLine={false} axisLine={false} domain={[0, 1]} />
              <Tooltip contentStyle={tooltipStyle} cursor={{ fill: 'rgba(255,255,255,0.03)' }} />
              <Legend wrapperStyle={{ fontSize: 12, paddingTop: 8 }} />
              <Bar isAnimationActive={false} dataKey="fBetaW" name="Fβw" fill="#10b981" radius={[3, 3, 0, 0]} />
              <Bar isAnimationActive={false} dataKey="mae"    name="MAE" fill="#ef4444" radius={[3, 3, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </section>
      </div>
    </div>
  );
}
