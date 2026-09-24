import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, NavLink, useLocation } from 'react-router-dom';
import axios from 'axios';
import TacticalViewport from './components/TacticalViewport';
import Table6Studio from './components/Table6Studio';
import DiagnosticsViewer from './components/DiagnosticsViewer';
import { Crosshair } from 'lucide-react';
import { API_BASE, AI_BASE } from './config';

const navItems = [
  { path: '/', label: 'Inference' },
  { path: '/table6', label: 'Results' },
  { path: '/diagnostics', label: 'Diagnostics' },
];

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

const statusColor = {
  online: 'var(--color-tactical-emerald)',
  offline: 'var(--color-tactical-crimson)',
  checking: 'var(--color-tactical-amber)',
};

function StatusDot({ label, status }) {
  return (
    <span className="flex items-center gap-1.5 text-xs" style={{ color: 'var(--text-faint)' }} title={`${label}: ${status}`}>
      <span className="w-1.5 h-1.5 rounded-full" style={{ background: statusColor[status] }} />
      {label}
    </span>
  );
}

function AppShell() {
  const location = useLocation();
  const serverStatus = useServiceStatus(`${API_BASE}/api/experiments`);
  const aiStatus = useServiceStatus(`${AI_BASE}/health`);

  return (
    <div className="min-h-screen flex flex-col">
      <header style={{ borderBottom: '1px solid var(--border)' }}>
        <div className="max-w-7xl mx-auto px-4 sm:px-6 h-14 flex items-center gap-6">
          <div className="flex items-center gap-2 shrink-0">
            <Crosshair className="w-5 h-5" style={{ color: 'var(--color-tactical-emerald)' }} />
            <span className="font-semibold tracking-tight">Ghost-Recon-CV</span>
          </div>

          <nav className="flex items-center gap-1 overflow-x-auto">
            {navItems.map((item) => (
              <NavLink
                key={item.path}
                to={item.path}
                end
                className="px-3 py-1.5 rounded-md text-sm whitespace-nowrap"
                style={({ isActive }) => ({
                  color: isActive ? 'var(--text)' : 'var(--text-faint)',
                  background: isActive ? '#1a222d' : 'transparent',
                })}
              >
                {item.label}
              </NavLink>
            ))}
          </nav>

          <div className="ml-auto hidden sm:flex items-center gap-4">
            <StatusDot label="Gateway" status={serverStatus} />
            <StatusDot label="Model" status={aiStatus} />
          </div>
        </div>
      </header>

      <main className="flex-grow">
        <div key={location.pathname} className="route-fade max-w-7xl mx-auto px-4 sm:px-6 py-6">
          <Routes>
            <Route path="/" element={<TacticalViewport />} />
            <Route path="/table6" element={<Table6Studio />} />
            <Route path="/diagnostics" element={<DiagnosticsViewer />} />
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
