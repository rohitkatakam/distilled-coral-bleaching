import React from 'react';
import { Github, FileText } from 'lucide-react';

export const Footer: React.FC = () => {
  return (
    <footer className="bg-gray-100 text-gray-700 py-12 border-t border-gray-300">
      <div className="max-w-4xl mx-auto px-4">
        {/* Main footer content */}
        <div className="text-center mb-8">
          <h3 className="text-lg font-bold text-black mb-2">Distilled Coral Bleaching</h3>
          <p className="text-sm text-gray-600 mb-1">Knowledge Distillation for Coral Bleaching Classification</p>
          <p className="text-sm text-gray-600">Rohit Katakam &middot; December 2025</p>
        </div>

        {/* Links */}
        <div className="flex flex-wrap items-center justify-center gap-6 mb-8">
          <a
            href="/earth_paper.pdf"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-2 text-black hover:text-gray-600 transition-colors font-medium"
          >
            <FileText size={18} />
            <span>Full Paper (PDF)</span>
          </a>
          <a
            href="https://github.com/rohitkatakam/distilled-coral-bleaching"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-2 text-black hover:text-gray-600 transition-colors font-medium"
          >
            <Github size={18} />
            <span>GitHub Repository</span>
          </a>
        </div>

        {/* AI Acknowledgment */}
        <div className="text-center text-xs text-gray-500 border-t border-gray-300 pt-6">
          <p className="mb-2">
            <strong>AI Acknowledgment:</strong> This project used AI for research, code debugging, LaTeX formatting, and proofreading.
          </p>
          <p>&copy; {new Date().getFullYear()} Rohit Katakam</p>
        </div>
      </div>
    </footer>
  );
};
