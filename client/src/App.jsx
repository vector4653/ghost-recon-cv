import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Link, useLocation } from 'react-router-dom';
import axios from 'axios';
import TacticalViewport from './components/TacticalViewport';
import Table6Studio from './components/Table6Studio';
import PaperReader from './components/PaperReader';
import DiagnosticsViewer from './components/DiagnosticsViewer';
import PresentationDeck from './components/PresentationDeck';
import { Target, BarChart2, Activity, Presentation, FileText, Crosshair } from 'lucide-react';
import { API_BASE, AI_BASE } from './config';

const navItems = [
  { path: '/', label: 'HUD Viewport', icon: Target },
  { path: '/table6', label: 'Table 6 Studio', icon: BarChart2 },
  { path: '/diagnostics', label: 'Diagnostics', icon: Activity },
  { path: '/deck', label: 'Slide Deck', icon: Presentation },
  { path: '/paper', label: 'IEEE Paper', icon: FileText },
];

function Navigation() {
  const location = useLocation();

  return (
    <nav className="glass-panel mx-6 mt-4 mb-2 p-1.5 flex flex-wrap justify-center items-center gap-1 relative">
      {navItems.map((item) => {
        const Icon = item.icon;
        const isActive = location.pathname === item.path;
        return (
          <Link
            key={item.path}
            to={item.path}
            className="relative flex items-center gap-2 px-4 py-2 rounded text-xs sm:text-sm uppercase tracking-wider"
            style={{
              color: isActive ? 'var(--color-tactical-emerald)' : '#94a3b8',
              background: isActive ? 'rgba(16,185,129,0.12)' : 'transparent',
              border: isActive ? '1px solid rgba(16,185,129,0.3)' : '1px solid transparent',
            }}
            onMouseEnter={(e) => { if (!isActive) e.currentTarget.style.color = '#e2e8f0'; }}
            onMouseLeave={(e) => { if (!isActive) e.currentTarget.style.color = '#94a3b8'; }}
          >
            <Icon className="w-4 h-4" /> {item.label}
            {isActive && (
              <span
                className="absolute left-3 right-3 -bottom-[7px] h-[2px] rounded-full"
                style={{ background: 'var(--color-tactical-emerald)', boxShadow: '0 0 6px var(--color-tactical-emerald)' }}
              />
            )}
          </Link>
        );
      })}
    </nav>
  );
}

function useServiceStatus(url, intervalMs = 15000) {
  const [status, setStatus] = useState('checking');

  useEffect(() => {
    let cancelled = false;
    const ping = async () => {
      try {
        await axios.get(url, { timeout: 4000 });
        if (!cancelled) setStatus('online');
      } catch {
        if (!cancelled) setStatus('offline');
      }
    };
    ping();
    const id = setInterval(ping, intervalMs);
    return () => { cancelled = true; clearInterval(id); };
  }, [url, intervalMs]);

  return status;
}

function StatusPill({ label, status }) {
  const colorMap = {
    online: 'var(--color-tactical-emerald)',
    offline: 'var(--color-tactical-crimson)',
    checking: 'var(--color-tactical-amber)',
  };
  const textMap = { online: 'ONLINE', offline: 'OFFLINE', checking: 'CHECKING' };
  const color = colorMap[status];

  return (
    <div
      className="flex items-center gap-1.5 px-2.5 py-1 rounded-full"
      style={{ background: 'rgba(0,0,0,0.3)', border: `1px solid ${color}44` }}
      title={`${label}: ${textMap[status]}`}
    >
      <span
        className={status !== 'offline' ? 'pulse-dot' : ''}
        style={{ width: 6, height: 6, borderRadius: '50%', background: color, boxShadow: `0 0 6px ${color}` }}
      />
      <span className="text-[10px] uppercase tracking-widest hidden sm:inline" style={{ color: '#94a3b8' }}>{label}</span>
      <span className="text-[10px] uppercase tracking-widest font-bold" style={{ color }}>{textMap[status]}</span>
    </div>
  );
}

function AppShell() {
  const location = useLocation();
  const serverStatus = useServiceStatus(`${API_BASE}/api/experiments`);
  const aiStatus = useServiceStatus(`${AI_BASE}/health`);

  return (
    <div className="min-h-screen flex flex-col">
      <header className="px-6 pt-5 pb-0 flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <div className="relative flex items-center justify-center w-8 h-8">
            <Crosshair className="w-6 h-6" style={{ color: 'var(--color-tactical-emerald)' }} />
            <div
              className="absolute inset-0 rounded-full pulse-dot"
              style={{ boxShadow: '0 0 14px rgba(16,185,129,0.6)' }}
            />
          </div>
          <div>
            <p className="text-sm sm:text-base font-bold uppercase tracking-widest leading-none" style={{ color: 'var(--color-tactical-emerald)' }}>
              Ghost-Recon-CV
            </p>
            <p className="text-[10px] uppercase tracking-widest mt-1" style={{ color: '#475569' }}>
              COD Tactical Intelligence Platform
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <StatusPill label="Gateway" status={serverStatus} />
          <StatusPill label="AI Sidecar" status={aiStatus} />
          <span className="text-xs hidden md:inline ml-2" style={{ color: '#334155' }}>
            Hwang &amp; Ma (2024) — Phase 4 Deployment
          </span>
        </div>
      </header>

      <Navigation />

      <main className="flex-grow">
        <div key={location.pathname} className="route-fade">
          <Routes>
            <Route path="/" element={<TacticalViewport />} />
            <Route path="/table6" element={<Table6Studio />} />
            <Route path="/diagnostics" element={<DiagnosticsViewer />} />
            <Route path="/deck" element={<PresentationDeck />} />
            <Route path="/paper" element={<PaperReader />} />
          </Routes>
        </div>
      </main>
    </div>
  );
}

export default function App() {
  return (
    <Router>
      <AppShell />
    </Router>
  );
}
