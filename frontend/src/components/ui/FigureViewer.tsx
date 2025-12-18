import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ZoomIn, X } from 'lucide-react';

interface FigureViewerProps {
  src: string;
  alt: string;
  caption?: string;
  className?: string;
}

export const FigureViewer: React.FC<FigureViewerProps> = ({ src, alt, caption, className = '' }) => {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <>
      <div
        className={`group relative cursor-zoom-in overflow-hidden ${className}`}
        onClick={() => setIsOpen(true)}
      >
        <img src={src} alt={alt} className="w-full h-auto object-cover transition-transform duration-300 group-hover:scale-102" />
        <div className="absolute inset-0 bg-black/0 group-hover:bg-black/5 transition-colors duration-300 flex items-center justify-center opacity-0 group-hover:opacity-100">
          <ZoomIn className="text-black w-10 h-10" />
        </div>
      </div>
      {caption && (
        <div className="mt-3 text-sm text-gray-600 text-center">
          {caption}
        </div>
      )}

      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/90 backdrop-blur-sm"
            onClick={() => setIsOpen(false)}
          >
            <motion.div
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.9, opacity: 0 }}
              className="relative max-w-7xl max-h-[90vh] overflow-auto rounded-lg"
              onClick={(e) => e.stopPropagation()}
            >
              <button 
                onClick={() => setIsOpen(false)}
                className="absolute top-4 right-4 p-2 bg-white/10 hover:bg-white/20 rounded-full text-white transition-colors z-10"
              >
                <X size={24} />
              </button>
              <img src={src} alt={alt} className="w-full h-auto rounded-lg shadow-2xl" />
              {caption && (
                <div className="mt-4 text-center text-white/90 font-medium text-lg">
                  {caption}
                </div>
              )}
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
};
