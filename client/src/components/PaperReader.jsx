import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { BookOpen, AlertCircle } from 'lucide-react';
import { API_BASE } from '../config';

export default function PaperReader() {
  const [content, setContent] = useState('Loading research paper...');

  useEffect(() => {
    axios.get(`${API_BASE}/api/paper`)
      .then(res => setContent(res.data.content))
      .catch(() => {
        setContent('# Camouflaged Object Detection (COD) in Tactical & Military Settings\\n\\n## Deep Learning Case Study & Replication of Hwang & Ma (2024)\\n\\n### Abstract\\nThis paper investigates military camouflaged target segmentation using deep learning. We adapt SINet-V2 to evaluate domain transfer between natural camouflage and artificial military camouflage. By replicating and extending the 6-experiment combination methodology, we assess how multi-domain camouflage training impacts soldier detection in tactical environments, specifically addressing the OODA loop constraints.\\n\\n[Please start the backend server to load the full paper]');
      });
  }, []);

  return (
    <div className="p-6 max-w-5xl mx-auto h-[85vh] overflow-y-auto custom-scrollbar">
      <div className="glass-panel p-8">
        <h1 className="text-2xl font-bold text-tactical-cyan uppercase tracking-wider mb-6 flex items-center gap-3 border-b border-slate-700 pb-4">
          <BookOpen className="w-6 h-6" /> IEEE Research Report
        </h1>
        
        <div className="prose prose-invert prose-slate max-w-none">
          {content.split('\\n').map((paragraph, idx) => (
            <p key={idx} className="mb-4 text-slate-300 leading-relaxed text-justify">
              {paragraph}
            </p>
          ))}
        </div>
      </div>
    </div>
  );
}
