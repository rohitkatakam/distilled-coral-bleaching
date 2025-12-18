import React from 'react';
import { Github, Download } from 'lucide-react';

export const Footer: React.FC = () => {
  return (
    <footer className="bg-ocean-950 text-ocean-200 py-12 border-t border-ocean-900">
      <div className="max-w-7xl mx-auto px-4 flex flex-col md:flex-row items-center justify-between gap-6">
        
        <div className="text-center md:text-left">
          <h3 className="text-xl font-bold text-white mb-2">Distilled Coral Bleaching</h3>
          <p className="text-sm text-ocean-400">Efficient AI for Marine Conservation</p>
        </div>

        <div className="flex items-center gap-6">
          <a 
            href="https://github.com/rohitkatakam/distilled-coral-bleaching" 
            target="_blank" 
            rel="noopener noreferrer"
            className="flex items-center gap-2 hover:text-white transition-colors"
          >
            <Github size={20} />
            <span>GitHub</span>
          </a>
          <a 
            href="/assets/earth_paper.pdf" 
            target="_blank"
            className="flex items-center gap-2 hover:text-white transition-colors"
          >
            <Download size={20} />
            <span>Download Paper</span>
          </a>
        </div>

        <div className="text-sm text-ocean-500">
          &copy; {new Date().getFullYear()} Project Team
        </div>
      </div>
    </footer>
  );
};
